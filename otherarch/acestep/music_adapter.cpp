#include "model_adapter.h"
#include "otherarch/utils.h"

#include <cmath>
#include <fstream>
#include <cstdio>
#include <regex>
#include <string>
#include <thread>
#include <vector>
#include <cstring>
#include <mutex>
#include <cinttypes>

#include "./request.cpp"
#include "./ace-qwen3.cpp"

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static int musicdebugmode = 0;
static bool music_is_quiet = false;
static bool musicgen_loaded = false;
static std::string musicvulkandeviceenv;

bool musictype_load_model(const music_load_model_inputs inputs)
{
    music_is_quiet = inputs.quiet;

    //duplicated from expose.cpp
    std::string vulkan_info_raw = inputs.vulkan_info;
    std::string vulkan_info_str = "";
    for (size_t i = 0; i < vulkan_info_raw.length(); ++i) {
        vulkan_info_str += vulkan_info_raw[i];
        if (i < vulkan_info_raw.length() - 1) {
            vulkan_info_str += ",";
        }
    }
    const char* existingenv = getenv("GGML_VK_VISIBLE_DEVICES");
    if(!existingenv && vulkan_info_str!="")
    {
        musicvulkandeviceenv = "GGML_VK_VISIBLE_DEVICES="+vulkan_info_str;
        putenv((char*)musicvulkandeviceenv.c_str());
    }

    std::string musicllm_filename = inputs.musicllm_filename;
    std::string musicembedding_filename = inputs.musicembedding_filename;
    std::string musicdiffusion_filename = inputs.musicdiffusion_filename;
    std::string musicvae_filename = inputs.musicvae_filename;
    printf("\nLoading Music Gen LLM Model: %s\nLoading Music Gen Embed Model: %s\nLoading Music Gen Diffusion Model: %s\nLoading Music Gen VAE Model: %s\n",
    musicllm_filename.c_str(),musicembedding_filename.c_str(),musicdiffusion_filename.c_str(),musicvae_filename.c_str());
    musicdebugmode = inputs.debugmode;

    bool ok = load_acestep(musicllm_filename);
    if (!ok) {
        printf("\nFailed to load Music Gen Model!\n");
        return false;
    }

    musicgen_loaded = true;

    printf("\nMusic Gen Load Complete.\n");
    return true;
}

music_generation_outputs musictype_generate(const music_generation_inputs inputs)
{
    music_generation_outputs output;

    if(!musicgen_loaded)
    {
        printf("\nWarning: KCPP music gen not initialized!\n");
        output.status = 0;
        return output;
    }

    if(!music_is_quiet)
    {
        printf("\nMusic Gen Generating...");
    }

    output.status = 1;
    return output;
}
