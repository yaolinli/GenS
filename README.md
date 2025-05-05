# GenS: Generative Frame Sampler for Long Video Understanding


<p align="center">
🔗 <a href="https://generative-sampler.github.io/" target="_blank">Project Page</a> · 📖 <a href="https://arxiv.org/abs/2503.09146" target="_blank">Paper</a> · ⭐ <a href="https://github.com/yaolinli/GenS" target="_blank">GitHub</a> · 📊 <a href="https://huggingface.co/datasets/yaolily/GenS-Video-150K" target="_blank">Dataset</a> · 🤗 <a href="https://huggingface.co/yaolily/GenS" target="_blank">Checkpoints</a>
</p>

## 📰 News

- **[2025-04-30]** We open-sourced GenS(Aria-based) model, code, and dataset! Try it in your long video QA projects requiring fewer but more informative frames.

- **[2025-03-08]** Our paper "Generative Frame Sampler for Long Video Understanding" is now available on arXiv.

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
```bash
conda create -n gens python=3.11
conda activate gens
pip install transformers==4.45.0 accelerate==0.34.1 sentencepiece==0.2.0 torchvision requests torch Pillow
pip install flash-attn --no-build-isolation
```

### Usage

#### Example Inference

```bash
# Using default video case in the video_example folder
python inference.py
```

**Output Format:**
The model returns relevance scores for frames in JSON format.
Example output: `{"11-12": 5, "16-21": 4, "28-30": 4, "46-49": 4, "22-27": 3, "33": 2}` means frame indexing 11-12 (i.e., sec011.png, sec012.png in the video_example folder) has highest relevance score 5.



#### Customized Inference
You can use the script with command-line arguments to customize your video and query:

```bash
python inference.py --model_id "yaolily/GenS" --video_path "path/to/your/video/frames" --question your_question
```

**Command-line Arguments**

- `--model_id`: HuggingFace model ID (default: "yaolily/GenS")
- `--video_path`: Directory containing video frame images (default: "video_example")
- `--question`: Question to ask about the video (default: "After styling the lady's hair, what action did the maid perform next?")



#### Programmatic Usage

You can also use GenS programmatically in your Python code:

```python
import glob
import os
from inference import setup_model, gens_frame_sampler

# Load model components
model_id = "yaolily/GenS" 
model, tokenizer, processor = setup_model(model_id)

# Load video frames
video_dir = "path/to/video/frames"
frame_paths = glob.glob(os.path.join(video_dir, "*.png"))  # or *.jpg, etc.
frame_paths.sort(key=lambda x: int(os.path.basename(x).split('sec')[1].split('.')[0]))

# Ask a question about the video
question = "What is happening in the kitchen scene?"

# Get frame relevance scores
result = gens_frame_sampler(question, frame_paths, model, tokenizer, processor)

# Process the results
print(f"Video: {video_dir}")
print(f"Question: {question}")
print(f"Relevant frames with scores: {result}")
```


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
