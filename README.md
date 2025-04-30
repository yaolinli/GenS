# GenS: Generative Frame Sampler for Long Video Understanding


<p align="center">
🔗 <a href="https://generative-sampler.github.io/" target="_blank">Project Page</a> · 📖 <a href="https://arxiv.org/abs/2503.09146" target="_blank">Paper</a> · ⭐ <a href="https://github.com/yaolinli/GenS" target="_blank">GitHub</a> · 📊 <a href="https://huggingface.co/datasets/yaolily/GenS-Video-150K" target="_blank">Dataset</a> · 🤗 <a href="https://huggingface.co/yaolily/GenS" target="_blank">Checkpoints</a>
</p>
📰 **News**

[2025-04-30] We open-sourced GenS(Aria-based) model, code, and dataset! Try it in your long video QA projects requiring fewer but more informative frame sampling.

[2025-03-08] Our paper "Generative Frame Sampler for Long Video Understanding" is now available on arXiv.

## Introduction

**GenS** (Generative Frame Sampler) is a novel approach that identifies question-relevant frames from long videos spanning minutes to hours. Given a long video and a user question, GenS effectively searches through the original massive collection of frames to produce a concise selection and enhances the performance of downstream VideoQA Assistants (such as Qwen2-VL, LLaVA-Video, VILA-v1.5, and Aria) by providing fewer but more informative frames.

GenS is built upon advanced long-context VideoLLMs (such as Aria and Qwen2.5VL), transforming key frame sampling into a generative task. 

<img src="https://generative-sampler.github.io/static/images/teaser.png" alt="GenS Framework" style="width: 100%;">

## Key Features of GenS

✨ **Temporal Understanding:**
GenS effectively captures temporal relationships between successive frames, enabling complex reasoning about temporal sequences such as "immediately after" events in videos.

📝 **Complex Instruction Understanding:**
Powered by built-in LLMs, GenS comprehends complex and flexible textual instructions, allowing it to interpret nuanced queries and identify the most relevant visual content.

⚡ **Effective Video-Text Alignment:**
Its native multi-modal architecture enables sophisticated multi-hop reasoning by seamlessly aligning long-range temporal cues with language semantics, resulting in more accurate frame selection.

🎉 **State-of-the-Art Performance:**
GenS significantly boosts the performance of various VideoQA models, achieving SOTA results on long-form video benchmarks when integrated with open-source models.

## Performance Highlights
-   🏆 **LongVideoBench**: LLaVA-Video-72B w/ GenS achieves **66.8** accuracy (+4.3)
-   🏆 **MLVU**: LLaVA-Video-72B w/ GenS achieves **77.0** accuracy (+2.7)
-   🏆 **HourVideo**: Aria w/ GenS obtains **39.2** accuracy, while Gemini-1.5-pro w/ GenS obtains **40.7** accuracy


<img src="https://generative-sampler.github.io/static/images/table_main.png" alt="Main Results Table" style="width: 100%;">
<img src="https://generative-sampler.github.io/static/images/hourvideo.png" alt="HourVideo Results Table" style="width: 100%;">

## Quick Start

### Installation
```
conda create -n gens python=3.10
conda activate gens
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.45.0 accelerate==0.34.1 sentencepiece==0.2.0 requests Pillow
pip install flash-attn --no-build-isolation
```

### Usage
```
python inference.py
```
### Example with Custom Video and Question

You can specify your own video frames path and question as follows:
```
# Example usage
if __name__ == "__main__":
    # Load model components
    model, tokenizer, processor = setup_model()
    
    # Example video frames (replace with your actual paths)
    frame_paths = [
        "/path/to/video/frames/00001.jpg", 
        "/path/to/video/frames/00002.jpg",
        # Add more frames...
    ]
    
    # Example question
    question = "Which frames show a person opening the door?"
    
    # Get frame relevance scores
    result = gens_frame_sampler(question, frame_paths, model, tokenizer, processor)
    
    print(f"Question: {question}")
    print(f"Relevant frames with scores: {result}")
```
**Output Format:**
The model returns relevance scores for frames in JSON format
Example output: `{"15": 5, "16": 4, "45-46": 3, ...}` means frame indexing 15 has relevance score 5, frame indexing 16 has relevance score 4, frame indexing 45-46 has relevance score 3, ...



## Citation
If you find our work helpful, please consider citing.
```
@article{yao2025gens,
    title={Generative Frame Sampler for Long Video Understanding},
    author={Yao, Linli and Wu, Haoning and Ouyang, Kun and Zhang, Yuanxing and Xiong, Caiming and Chen, Bei and Sun, Xu and Li, Junnan},
    journal={arXiv preprint arXiv:2503.09146},
    year={2025}
}
```