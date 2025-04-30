import torch
from PIL import Image
import sys
import os
from typing import List

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


def setup_model():
    """Set up and load the GenS model and its components."""
    
    # Register custom models with the Auto classes
    AutoConfig.register("yi_vl", YiVLConfig)
    AutoModel.register(YiVLConfig, YiVLForConditionalGeneration)
    AutoConfig.register("deepseek", DeepseekConfig)
    AutoModelForCausalLM.register(DeepseekConfig, DeepseekForCausalLM)
    
    # Load model from Hugging Face
    model_id = "yaolily/GenS"
    
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
    
    print("GenS Model loaded successfully!")
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
    
    # Process the images and text
    inputs = processor(
        text=[formatted_question],
        images=frames,
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0
        )
    
    # Decode and extract the relevant part of the response
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    result = response.split("assistant\n")[-1].split("<|im_end|>")[0].strip()
    
    return result


# Example usage
if __name__ == "__main__":
    # Load model components
    model, tokenizer, processor = setup_model()
    
    # Example video frames (replace with your actual paths)
    example_video_path = "video_example"
    frame_paths = glob.glob(os.path.join(example_video_path, "*.png"))
    print(f"Input {len(frame_paths)} frames")
    # Example question
    question = "After styling the lady's hair, what action did the maid perform next?"
    
    # Get frame relevance scores
    result = gens_frame_sampler(question, frame_paths, model, tokenizer, processor)
    
    print(f"Question: {question}")
    print(f"Relevant frames with scores: {result}")