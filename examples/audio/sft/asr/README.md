# Results

see [LlamaForASR](https://github.com/xingchensong/TouchNet/blob/main/docs/LlamaForASR.md) for a brief introduction.

We only reported the results on wenetspeech (~10000h), and did not report on aishell (~100h) / librispeech (~1000h). This is because these two datasets are too small. Even a small model with 50M parameters is very easy to overfit. However, on the wenetspeech dataset, when we scale the model parameters from 50M to 1B, we still did not observe the overfiting phenomenon. 10000 hours might be a better dataset standard for observation and learning scaling.
