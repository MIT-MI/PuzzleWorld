# PuzzleWorld: A Benchmark for Multimodal, Open-Ended Reasoning in Puzzlehunts

[![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace-Dataset-orange?logo=huggingface)](https://huggingface.co/datasets/hzli1202/PuzzleWorld)
[![arXiv](https://img.shields.io/badge/arXiv-2506.06211-b31b1b.svg)](https://arxiv.org/abs/2506.06211)

PuzzleWorld is a benchmark of 667 real-world puzzlehunt–style problems designed to evaluate open-ended, multimodal reasoning capabilities of AI models. Curated from Puzzled Pint’s Creative Commons–licensed archives (2010–2025), each puzzle combines text, visual, and structured inputs with no explicitly stated instructions. Solvers must first infer the hidden problem structure from ambiguous clues and then execute a multi-step, creative reasoning chain to arrive at a short, canonical answer. Each puzzle is accompanied by detailed, human-annotated reasoning traces, labeled with required cognitive skills (e.g., logic, wordplay, cryptic decoding, spatial reasoning, knowledge, commonsense). PuzzleWorld also provides per-puzzle metadata (title, flavor text, difficulty, input modalities, skills, source URL) and associated puzzle images. 

## 📂 Dataset Access

The full dataset is hosted on Hugging Face:
👉 [PuzzleWorld on Hugging Face](https://huggingface.co/datasets/hzli1202/PuzzleWorld)

Puzzlehunts are a genre of complex, multi-step puzzles lacking well-defined problem definitions. In contrast to conventional reasoning benchmarks consisting of tasks with clear instructions, puzzlehunts require models to discover the underlying problem structure from multimodal evidence and iterative reasoning, mirroring real-world domains such as scientific discovery, exploratory data analysis, or investigative problem-solving. Despite recent progress in foundation models, their performance on such open-ended settings remains largely untested. In this paper, we introduce PuzzleWorld, a large-scale benchmark of 667 puzzlehunt-style problems designed to assess step-by-step, open-ended, and creative multimodal reasoning. Each puzzle is annotated with the final solution, detailed reasoning traces, and cognitive skill labels, enabling holistic benchmarking and fine-grained diagnostic analysis. Most state-of-the-art models achieve only 1-2% final answer accuracy, with the best model solving only 14% of puzzles and reaching 40% stepwise accuracy. To demonstrate the value of our reasoning annotations, we show that fine-tuning a small model on reasoning traces improves stepwise reasoning from 4% to 11%, while training on final answers alone degrades performance to near zero. Our error analysis reveals that current models exhibit myopic reasoning, are bottlenecked by the limitations of language-based inference, and lack sketching capabilities crucial for visual and spatial reasoning. 

## 🔗 Resources
- [Paper](https://arxiv.org/abs/2506.06211)
- [Data](https://huggingface.co/datasets/hzli1202/PuzzleWorld)
- [Data Source](https://puzzledpint.org)

## 📑 Citation
If you use PuzzleWorld in your research, please cite our paper ❤️ 

```bibtex
@article{li2025puzzleworld,
  title={PuzzleWorld: A Benchmark for Multimodal, Open-Ended Reasoning in Puzzlehunts},
  author={Li, Hengzhi and Jiang, Brendon and Naehu, Alexander and Song, Regan and Zhang, Justin and Tjandrasuwita, Megan and Ekbote, Chanakya and Chen, Steven-Shine and Balachandran, Adithya and Dai, Wei and others},
  journal={arXiv preprint arXiv:2506.06211},
  year={2025}
}
```
