# Results

We provide a simple comparison experiment here: The difference in behavior between randomly initialized pretraining (`fromscratch`) and pretraining initialized with llama3.2-1B weights (`frompretrain`) on the c4 dataset under the same training configuration (learning rate, eps, llama3.2-1B model structure, etc.).

According to the training log shown in the figure below, the dev loss of the `frompretrain` method is significantly better than that of `fromscratch` at the beginning of training. As training progresses, the `frompretrain` method is always continuously better than `fromscratch`.

![Image](https://github.com/user-attachments/assets/ff63f935-d899-4976-8060-e8cd3c403758)
