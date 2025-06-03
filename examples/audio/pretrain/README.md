# Results

As for audio pretraining, we adopt the same model structure as in [TouchAudioForCausalLM](https://github.com/xingchensong/TouchNet/blob/main/docs/TouchAudioForCausalLM.md).
We use a training-free audio tokenizer called `BEST-RQ`, which uses a randomly initialized codebook and projector to quantize the input audio. Although its method is very simple, it shows impressive results on speech SSL.

<div align="center">

![touchnet_pretrain](https://github.com/xingchensong/TouchNet/blob/main/assets/touchnet_pretrain.png)

Comparison between [BEST-RQ](https://arxiv.org/pdf/2202.01855), [NEST-RQ](https://arxiv.org/abs/2409.08680) and our proposed `👆 TouchNet` solution.

</div>

Regarding the optimization of the training process, we have made four improvements on the basis of `BEST-RQ`:

1. **`Next-Token Prediction is ALL YOU NEED`**: We adopt similar NTP behavior that has been proposed in `NEST-RQ`. `NEST-RQ` uses the same training-free tokenizer as `BEST-RQ` but replaces the BERT-style mask with GPT-style mask. It achieves similar or even better performance compared to `BEST-RQ`. Given that the primary design philosophy of TouchNet is that `all designs must give way to scaling` and NTP has been proven effective in scaling, we prefer `NEST-RQ` over `BEST-RQ` here.
2. **`Convolution-free model arch`**: To scale more efficiently and easily, we further simplify the model structure and training process on the basis of NTP of `NEST-RQ`. For example, here we simply replace the convolutional downsampling module with a linear projector. At the same time, the main structure of the model follows the LLM design. This convolution-free structure enables us to utilize the existing ecosystem more easily. For more discussion about this model architecture, see [TouchAudioForCausalLM](https://github.com/xingchensong/TouchNet/blob/main/docs/TouchAudioForCausalLM.md).
3. **`Decouple the BEST-RQ tokenization process and the model forward process`**: In addition to simplifying the model structure, we also simplify the training process. As shown in the figure below, previous methods like `BEST-RQ` and `NEST-EQ` couple the tokenizer and the main body of the model. The random projection and Argmin in the speech quantization process run on the GPU. This coupling is not friendly for large-scale parallel training (such as TP/PP). Thanks to the determinism in initialization and the simplicity in computation of the BEST-RQ tokenizer, we move all the tokenization processes to the CPU. By adding num_workers in the dataloader, we can simply achieve a good overlapping between CPU quantization and GPU model forwarding. Under the design of this structure, whether it is pure text input or multimodal input, it has uniformity for the N-D parallel training system, thereby better reusing the existing ecosystem.
4. **`CMVN-free`**: We neither use the global mean and var norm common in the speech field nor the sentence-level norm. Instead, we use a parameterless layer norm to perform feature dimension normalization on the stacked audio features. This is to reduce inductive bias and achieve better streaming adaptability.

Here we give a pretraining example on wenetspeech (~10000 hours). As shown in the figure below, for both the large model (1B in red) and the small model (50M in orange), the NTP loss curve is smooth. The final dev accuracy is around 35%. Besides, we also observe a good scaling pattern when scaling from the small model (50M) to the large model (1B).

![Image](https://github.com/user-attachments/assets/84b761cf-c26e-483c-876f-e55e054d5512)
