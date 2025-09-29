import torch
from diffusers import StableDiffusionPipeline


MODEL_CARDS = {
    # Stable-Diffusion
    'sd14': "CompVis/stable-diffusion-v1-4",
    # 'sd15': "runwayml/stable-diffusion-v1-5",
    'sd15': "stable-diffusion-v1-5/stable-diffusion-v1-5",
    'sd20b': "stabilityai/stable-diffusion-2-base",
    'sd20': "stabilityai/stable-diffusion-2",
    'sd21b': "stabilityai/stable-diffusion-2-1-base",
    'sd21': "stabilityai/stable-diffusion-2-1",
    # HumanNorm
    "normal-adapted": "xanderhuang/normal-adapted-sd1.5",
    "depth-adapted": "xanderhuang/depth-adapted-sd1.5",
    # Stable-Diffusion-XL
    'sdxl10': "stabilityai/stable-diffusion-xl-base-1.0",
    #Anime Style Diffusion
    'mistoon': "stablediffusionapi/mistoonanime",
    'counterfeit': "gsdf/counterfeit-v1.0",
    'animemix': "stablediffusionapi/aam-xl-anime-mix",
    'wd': "hakurei/waifu-diffusion",
    'disney': "stablediffusionapi/disney-pixar-cartoon",
    #Anime Lora
    'sd12': "Linaqruf/animagine-xl-2.0",
    'dl': "dreamlike-art/dreamlike-anime-1.0",
    '1': "NovelAI/nai-anime-v1-full",
    '2': "Ojimi/anime-kawai-diffusion",
    '3': "John6666/souu-anime-cute-pony-v10-sdxl",
    '4': "glif/90s-anime-art",
    '5': "iriscope/cartoonavatar",
    '6': "stablediffusionapi/anime-diffusion",
    '7': "FredZhang7/anime-anything-promptgen-v2",
    '8': "alea31415/onimai-characters",
    '9': "stablediffusionapi/illustration-art",
    '10': "stablediffusionapi/mistoonanime-v30",
    '11': "stablediffusionapi/anime-model-v2",
    '12': "stablediffusionapi/anime-journey",
    '13': "Linaqruf/animagine-xl",
    '14': "nitrosocke/Ghibli-Diffusion",
    '15': "hakurei/waifu-diffusion-xl",
    '16': "alvdansen/softserve_anime",
    '17': "Muinez/sana-512-anime"
}

# Function to measure memory usage of each model
def check_model_memory_usage(model_cards):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for model_key, model_name in model_cards.items():
        try:
            print(f"Loading model: {model_key} ({model_name})")
            # Load pipeline in float16 precision
            pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
            pipe.to(device)

            # Get memory usage
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # Convert to GB
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # Convert to GB

            results[model_key] = {
                "model_name": model_name,
                "memory_allocated_gb": memory_allocated,
                "memory_reserved_gb": memory_reserved,
            }
            print(f"Model: {model_key} | Allocated: {memory_allocated:.2f} GB | Reserved: {memory_reserved:.2f} GB\n")
        
        except Exception as e:
            results[model_key] = {"error": str(e)}
            print(f"Error loading model {model_key} ({model_name}): {e}\n")
        
        finally:
            # Clear CUDA cache to free memory
            torch.cuda.empty_cache()
    
    return results

# Run the memory check
if __name__ == "__main__":
    memory_usage_results = check_model_memory_usage(MODEL_CARDS)
    print("\n=== Memory Usage Results ===")
    for model_key, info in memory_usage_results.items():
        if "error" in info:
            print(f"{model_key}: Error - {info['error']}")
        else:
            print(f"{model_key}: Allocated: {info['memory_allocated_gb']:.2f} GB | Reserved: {info['memory_reserved_gb']:.2f} GB")