import torch
from PIL import Image
import sys
import os
import re
import glob
from typing import List, Dict, Union, Optional

# Import required libraries
from transformers import AutoProcessor, AutoTokenizer, AutoConfig, AutoModel, AutoModelForCausalLM
from yivl.yivl_model_hf import YiVLForConditionalGeneration, YiVLConfig
from yivl.siglip_navit_490 import NaViTProcessor
from yivl.constants import (
    DEFAULT_IMAGE_END_TOKEN,
    DEFAULT_IMAGE_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from deepseekv1moe.modeling_deepseek import DeepseekConfig, DeepseekForCausalLM


def load_and_process_image(image_paths: List[Union[str, Image.Image]], 
                          split_image: bool, 
                          split_ratio: List[List[int]], 
                          patch_size: int, 
                          image_processor, 
                          use_navit_preprocessor: bool, 
                          model) -> tuple:
    """
    Load and process images for Gens model.
    
    Args:
        image_paths: List of image paths or PIL images
        split_image: Whether to split images
        split_ratio: Ratio for splitting images
        patch_size: Size of image patches
        image_processor: The image processor
        use_navit_preprocessor: Whether to use NaViT preprocessor
        model: The model for config reference
        
    Returns:
        tuple: (image_tensor, pixel_attention_mask, image_crops)
    """
    images = []
    image_crops = []
    
    # Process each image
    for img_path in image_paths:
        # Handle PIL Image or file path
        if isinstance(img_path, Image.Image):
            pil_img = img_path.convert("RGB")
        else:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            with Image.open(img_path) as image:
                pil_img = image.convert("RGB")
        
        # Process image based on model requirements
        if getattr(model.config, "image_aspect_ratio", None) == "clip":
            pil_img = expand2square(
                pil_img, tuple(int(x * 255) for x in image_processor.image_mean)
            )
        elif getattr(model.config, "image_aspect_ratio", None) == "resize":
            pil_img = pil_img.resize((patch_size, patch_size))
        
        images.append(pil_img)
        image_crops.append(1)  # One crop per image
    
    # Process images with the appropriate processor
    pixel_attention_mask = None
    if use_navit_preprocessor:
        process_image_list = []
        navit_pixel_mask = []
        for single_image in images:
            return_dict = image_processor(
                image=single_image, image_max_size=patch_size
            )
            process_image_list.append(return_dict["image"])
            navit_pixel_mask.append(return_dict["pixel_mask"])
        image_tensor = torch.stack(process_image_list, dim=0)
        pixel_attention_mask = torch.stack(navit_pixel_mask, dim=0)
    else:
        image_tensor = image_processor.preprocess(
            images,
            size={"width": patch_size, "height": patch_size},
            return_tensors="pt",
        )["pixel_values"]
    
    return image_tensor, pixel_attention_mask, image_crops


def expand2square(pil_img, background_color):
    """Expand image to square with specified background color."""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def _tokenizer_image_token(
    config,
    tokenizer,
    prompt: str,
    input_image_tag: str = None,
    crop_nums: List[int] = None,
) -> torch.Tensor:
    """
    Tokenize the prompt, with special handling of the image token.
    
    :param config: The model config.
    :param tokenizer: The tokenizer using.
    :param prompt: The prompt to tokenize.
    :param input_image_tag: The image tag that represent a image in prompt.
    :returns: The list of tokens.
    """
    from yivl.constants import (
        DEFAULT_IMAGE_END_TOKEN,
        DEFAULT_IMAGE_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    
    if input_image_tag is not None:
        replace = DEFAULT_IMAGE_TOKEN
        if crop_nums is not None:
            # we had multiple cropped images
            for crops in crop_nums:
                if getattr(config, "mm_use_im_start_end", True):
                    replace = (
                        DEFAULT_IMAGE_START_TOKEN
                        + DEFAULT_IMAGE_TOKEN * crops
                        + DEFAULT_IMAGE_END_TOKEN
                    )
                prompt = re.sub(input_image_tag, replace, prompt, count=1)

    prompt_chunks = []
    for chunk in prompt.split(DEFAULT_IMAGE_TOKEN):
        chunk_enc = tokenizer(chunk).input_ids
        prompt_chunks.append(chunk_enc)

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(
        prompt_chunks, [IMAGE_TOKEN_INDEX] * (offset + 1)
    ):
        input_ids.extend(x[offset:])
    return torch.tensor(input_ids)


def setup_model(model_id="yaolily/GenS"):
    """Set up and load the GenS model and its components.
    
    Args:
        model_id: HuggingFace model ID to load
        
    Returns:
        tuple: (model, tokenizer, processor)
    """
    
    # Register custom models with the Auto classes
    AutoConfig.register("yi_vl", YiVLConfig)
    AutoModel.register(YiVLConfig, YiVLForConditionalGeneration)
    AutoConfig.register("deepseek", DeepseekConfig)
    AutoModelForCausalLM.register(DeepseekConfig, DeepseekForCausalLM)
    
    # Load configuration
    config = AutoConfig.from_pretrained(model_id)
    
    # Load model with optimizations
    model = AutoModel.from_pretrained(
        model_id,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16
    ).to(torch.device("cuda"))
    
    # Load tokenizer with special token handling
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    if not tokenizer.pad_token or tokenizer.pad_token_id < 0:
        try:
            tokenizer.add_special_tokens({"pad_token": "<unk>"})
            if tokenizer.pad_token_id is None:
                tokenizer.add_special_tokens({"pad_token": "<mask>"})
        except ValueError:
            tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
    
    # Initialize the custom image processor
    processor = NaViTProcessor(image_max_size=490)
    
    print(f"GenS Model '{model_id}' loaded successfully!")
    return model, tokenizer, processor


def gens_frame_sampler(question: str, frame_paths: List[str], model, tokenizer, processor):
    """
    Use GenS model to identify and score relevant frames for a video question.
    
    Args:
        question: The question to answer about the video
        frame_paths: List of paths to video frames
        model: Pre-loaded GenS model
        tokenizer: Pre-loaded tokenizer
        processor: Pre-loaded image processor
        
    Returns:
        The model's response with relevance scores for frames
    """
    # Load frames as PIL images
    frames = []
    for path in frame_paths:
        try:
            img = Image.open(path).convert("RGB")
            # Optional: resize images to expected size
            if img.width > 490 or img.height > 490:
                ratio = min(490/img.width, 490/img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size)
            frames.append(img)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
    
    if not frames:
        return "Error: No valid frames could be loaded"
    
    # Create prompt
    prompt = """Please identify the video frames most relevant to the given question and provide 
              their timestamps in seconds along with a relevance score. The score should be on a 
              scale from 1 to 5, where higher scores indicate greater relevance. Return the output 
              strictly in the following JSON format: {"timestamp": score, ...}."""
    
    # Format the input as expected by the model
    frm_placeholders = [f"[{i}]<image1>\n" for i in range(len(frames))]
    content = "{}Question: {}\n{}".format("".join(frm_placeholders), question, prompt)
    question_data = [{"role": "user", "content": content}]
    
    # Apply chat template
    formatted_question = tokenizer.apply_chat_template(question_data, add_generation_prompt=True, tokenize=False)
    
    try:
        # Process images
        images_tensor, pixel_attention_mask, num_crops = load_and_process_image(
            frames, False, [[1, 1]], 490, processor, True, model
        )
        
        # Tokenize with image tokens
        encoded_question = _tokenizer_image_token(model.config, tokenizer, formatted_question, r"<image\d>", num_crops)
        
        # Move tensors to device
        device = model.device
        encoded_question = encoded_question.unsqueeze(0).to(device)
        images_tensor = images_tensor.to(device, dtype=torch.bfloat16)
        if pixel_attention_mask is not None:
            pixel_attention_mask = pixel_attention_mask.to(device, dtype=torch.int64)
        
        # Model inference
        with torch.no_grad():
            outputs = model.generate(
                encoded_question,
                attention_mask=encoded_question.ne(tokenizer.pad_token_id),
                pixel_values=images_tensor,
                # Uncomment if needed:
                # pixel_attention_mask=pixel_attention_mask,
                temperature=0.0,
                max_new_tokens=256,
                use_cache=True,
            )
        
        # Decode and extract the relevant part of the response
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        result = response.split("assistant\n")[-1].split("<|im_end|>")[0].strip()
        
        return result
        
    except Exception as e:
        print(f"Error during frame sampling: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"


# Example usage
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run GenS frame sampling")
    parser.add_argument("--model_id", type=str, default="yaolily/GenS", help="HuggingFace model ID")
    parser.add_argument("--video_path", type=str, default="video_example", help="Path to video frames")
    parser.add_argument("--question", type=str, default="After styling the lady's hair, what action did the maid perform next?", 
                       help="Question to answer about the video")
    args = parser.parse_args()
    
    # Load model components
    model, tokenizer, processor = setup_model(args.model_id)
    
    # Example video frames
    frame_paths = glob.glob(os.path.join(args.video_path, "*.png"))
    if not frame_paths:
        print(f"No frames found in {args.video_path}")
        sys.exit(1)
        
    frame_paths.sort(key=lambda x: int(os.path.basename(x).split('sec')[1].split('.')[0]))
    print(f"Input {len(frame_paths)} frames")
    
    # Get frame relevance scores
    result = gens_frame_sampler(args.question, frame_paths, model, tokenizer, processor)
    
    print(f"Question: {args.question}")
    print(f"Relevant frames with scores: {result}")