# Results

We only reported the results on wenetspeech (~10000h), and did not report on aishell (~100h) / librispeech (~1000h). This is because these two datasets are too small. Even a small model with 50M parameters is very easy to overfit. However, on the wenetspeech dataset, when we scale the model parameters from 50M to 1B, we still did not observe the overfiting phenomenon. 10000 hours might be a better dataset standard for observation and learning scaling.


NOTE: We used the same WER calculation method as [SpeechIO](https://github.com/SpeechColab/Leaderboard/blob/master/utils/benchmark.sh#L54-L103).


## WenetSpeech

| exp id |   model   | note | instruct | Test\_Net | Test\_Meeting |
|:---:|:-------------------:|:----:|:---------:|:-------------:|:-----:|
| 0 |  Qwen2-Audio-7B-Base     | original model       |  Detect the language and recognize the speech: <\|zh\|>  | 8.15 [ 33422 / 409974, 8260 ins, 8070 del, 17092 sub ]  |    7.08 [ 15105 / 213221, 3280 ins, 3180 del, 8645 sub ]   |
| 1 |  Qwen2-Audio-7B-Base     | finetuned model      |  Generate the transcription:  | 5.58 [ 22858 / 409979, 2119 ins, 4287 del, 16452 sub ]  |    6.51 [ 13887 / 213221, 2324 ins, 3275 del, 8288 sub ]   |
| 2 |  Qwen2-Audio-7B-Base     | from scratch       |  Generate the transcription:  | 11.28 [ 46253 / 409981, 3187 ins, 7074 del, 35992 sub ]  | 18.90 [ 40303 / 213221, 4016 ins, 7210 del, 29077 sub ] |
| 3 |  Kimi-Audio-7B-Instruct    | original model      |  请把这段语音转录成文本。  |  4.91 [ 20134 / 409974, 2538 ins, 3348 del, 14248 sub ]  |  5.26 [ 11217 / 213221, 2085 ins, 2638 del, 6494 sub ]   |
| 4 |  Kimi-Audio-7B-Instruct    | original model      |  Generate the transcription:  |  6.86 [ 28137 / 409981, 9471 ins, 3219 del, 15447 sub ]  |  5.28 [ 11250 / 213221, 2081 ins, 2639 del, 6530 sub ]   |
| 5 |  Kimi-Audio-7B-Base    | original model      | N/A (Base model cannot do ASR task directly)  | N/A |  N/A  |
| 6 |  Kimi-Audio-7B-Base    | finetuned model      |  Generate the transcription:  | 5.63 [ 23086 / 409981, 2367 ins, 4148 del, 16571 sub ]  |     7.50 [ 15987 / 213221, 2435 ins, 4625 del, 8927 sub ]  |


1. Comparing 0 & 1, Finetuning Qwen2-Audio on wenetspeech training set get much better results.
2. Comparing 1 & 2, Pretraining benifits a lot on downstream task.
3. Comparing 3 & 4, Kimi-Audio-7B-Instruct is highly sensitive to prompts. If you change an instruction, the WER will deteriorate significantly.
4. Comparing 4 & 6, Finetuning Kimi-Audio-7B-Base achieved better Test_Net results (but worse Test_Meeting results) than Kimi-Audio-7B-Instruct **under the same instruction**.
5. Comparing 1 & 6, Although there is an order-of-magnitude difference in the pre-training data between Qwen2-Audio-7B and Kimi-Audio-7B-Base, their final performances are similar when fine-tuned using the same SFT data.
