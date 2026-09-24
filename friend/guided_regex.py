"""A bounded full-match regex subset compiled to native GBNF.

Character categories use ASCII semantics. Unsupported assertions, flags,
backreferences and ambiguous escapes raise ValueError before generation starts.
Rules are named instead of expanded inline to bound compiler memory use.
"""
import re


def compile_regex(pattern):
    if not isinstance(pattern, str) or not pattern or len(pattern) > 2048:
        raise ValueError("guided_regex must be a nonempty string of at most 2048 characters")
    # Check ranges and malformed syntax without ever matching user input.
    try:
        re.compile(pattern, re.ASCII)
    except (re.error, RecursionError) as exc:
        raise ValueError("invalid guided_regex") from exc
    return _Compiler(pattern).compile()


class _Compiler:
    categories = {"d": "0-9", "w": "A-Za-z0-9_", "s": r" \t\n\r\x0b\x0c"}

    def __init__(self, pattern):
        self.text = pattern
        self.pos = 0
        self.rules = []
        self.cost = 0

    @staticmethod
    def character(char):
        value = ord(char)
        if 0xd800 <= value <= 0xdfff:
            raise ValueError("surrogates are unsupported in guided_regex")
        # Hex escapes work both inside terminals and character classes, including
        # NUL, quote, backslash, hyphen and the class-negation character.
        return "\\x%02x" % value if value < 256 else "\\U%08x" % value

    def rule(self, body):
        self.cost += len(body)
        if self.cost > 32768 or len(self.rules) >= 1024:
            raise ValueError("guided_regex grammar is too large")
        name = "regex-%d" % len(self.rules)
        self.rules.append(name + " ::= " + body)
        return name

    def escape(self, in_class=False):
        if self.pos >= len(self.text):
            raise ValueError("trailing regex escape")
        char = self.text[self.pos]
        self.pos += 1
        if char in self.categories:
            return self.categories[char], True
        if char in "DWS":
            if in_class:
                raise ValueError("negated categories within classes are unsupported")
            return "[^" + self.categories[char.lower()] + "]", True
        if char in "nrtfv":
            return self.character({"n": "\n", "r": "\r", "t": "\t", "f": "\f", "v": "\v"}[char]), False
        if char in "xuU":
            size = {"x": 2, "u": 4, "U": 8}[char]
            digits = self.text[self.pos:self.pos + size]
            if len(digits) != size or not all(c in "0123456789abcdefABCDEF" for c in digits):
                raise ValueError("invalid hexadecimal escape")
            self.pos += size
            return self.character(chr(int(digits, 16))), False
        if char.isalnum():
            raise ValueError("backreferences, boundaries and named escapes are unsupported")
        return self.character(char), False

    def character_class(self):
        negate = self.pos < len(self.text) and self.text[self.pos] == "^"
        self.pos += int(negate)
        parts = []
        first = True
        while self.pos < len(self.text):
            char = self.text[self.pos]
            self.pos += 1
            if char == "]" and not first:
                return "[" + ("^" if negate else "") + "".join(parts) + "]"
            if char == "\\":
                part, _ = self.escape(in_class=True)
            elif char == "-" and not first and self.text[self.pos:self.pos + 1] != "]":
                part = "-"  # re.compile already validated the endpoints.
            else:
                part = self.character(char)
            parts.append(part)
            first = False
        raise ValueError("unterminated character class")

    def expression(self, depth=0):
        if depth > 24:
            raise ValueError("guided_regex nesting is too deep")
        alternatives, sequence = [], []
        nullable_alt, nullable_sequence = False, True
        while self.pos < len(self.text) and self.text[self.pos] != ")":
            char = self.text[self.pos]
            self.pos += 1
            if char == "|":
                alternatives.append(" ".join(sequence) or '""')
                nullable_alt |= nullable_sequence
                sequence, nullable_sequence = [], True
                continue
            nullable = False
            if char == "(":
                if self.text[self.pos:self.pos + 2] == "?:":
                    self.pos += 2
                elif self.text[self.pos:self.pos + 1] == "?":
                    raise ValueError("regex flags, lookarounds and named groups are unsupported")
                atom, nullable = self.expression(depth + 1)
                if self.text[self.pos:self.pos + 1] != ")":
                    raise ValueError("unterminated group")
                self.pos += 1
            elif char == "[":
                atom = self.character_class()
            elif char == ".":
                atom = r"[^\n]"
            elif char == "\\":
                escaped, category = self.escape()
                atom = escaped if escaped.startswith("[") and category else (
                    "[" + escaped + "]" if category else '"' + escaped + '"')
            elif char in "^$*+?{}":
                raise ValueError("unsupported anchor or misplaced quantifier")
            else:
                atom = '"' + self.character(char) + '"'

            quantifier = self.text[self.pos:self.pos + 1]
            minimum, maximum = 1, 1
            if quantifier and quantifier in "*+?":
                self.pos += 1
                minimum, maximum = {"*": (0, None), "+": (1, None), "?": (0, 1)}[quantifier]
            elif quantifier == "{":
                end = self.text.find("}", self.pos)
                if end < 0:
                    raise ValueError("unterminated repetition")
                bounds = self.text[self.pos + 1:end].split(",")
                if len(bounds) > 2 or not bounds[0].isdigit() or (len(bounds) == 2 and bounds[1] and not bounds[1].isdigit()):
                    raise ValueError("invalid repetition")
                minimum = int(bounds[0])
                maximum = minimum if len(bounds) == 1 else (int(bounds[1]) if bounds[1] else None)
                self.pos = end + 1
            if minimum > 64 or (maximum is not None and (maximum > 64 or maximum < minimum)):
                raise ValueError("repetition bounds must be within 0..64")
            if maximum is None and nullable:
                raise ValueError("unbounded repetition of an empty match is unsupported")
            if minimum != 1 or maximum != 1:
                name = self.rule(atom)
                if maximum is None:
                    atom = " ".join([name] * minimum + [name + "*"])
                else:
                    atom = " ".join([name] * minimum + [name + "?"] * (maximum - minimum)) or '""'
                atom = self.rule(atom)
                nullable |= minimum == 0
            sequence.append(atom)
            nullable_sequence &= nullable
        alternatives.append(" ".join(sequence) or '""')
        return self.rule(" | ".join(alternatives)), nullable_alt or nullable_sequence

    def compile(self):
        if self.text.startswith("^"):
            self.text = self.text[1:]
        # Strip only an unescaped final anchor; \\$ is a literal dollar sign.
        if self.text.endswith("$"):
            backslashes = len(self.text[:-1]) - len(self.text[:-1].rstrip("\\"))
            if backslashes % 2 == 0:
                self.text = self.text[:-1]
        root, _ = self.expression()
        if self.pos != len(self.text):
            raise ValueError("unmatched group")
        return "root ::= " + root + "\n" + "\n".join(self.rules) + "\n"
