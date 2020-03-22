# Transformers built-up from foundations
Building up the Transformer model family from scratch

### Motivation 
Modern Transformer-based models (BERT, GPT-2, XLNet, and T5 among others...) are all built on the same fundamental block, the self-attention, with differences in their architectures and training objectives, as well as other variations. Although there exist many well-documented open-source codes that implement these models, learners often face difficulties to understand and digest them due to their complexity. What makes things even more difficult is that in general these models are huge in their size, and pre-training these models on large-scale corpora is prohibitely expensive to most individuals. Of course, pre-trained language models are generously open-sourced and you can use them in the supposed way - transfer learning or finetuning. However, as a learner, I always get the best learning experience when implementing every fine details myself, from the very bottom. So I decided to begin building up the Transformer-based models in **a trainable way**: relatively simple and training on a small corpus or dataset. 

This repository is a recored of my own journey and little experiment to implement these modern transformer model family from the fundamentals - self-attention, tensor multiplication, and optionally with [Einstein summation convention](https://en.wikipedia.org/wiki/Einstein_notation) ("einsum") as a way of adopting explicitness. All of my codes are meant to run on "small-scale" systems, such as a personal computer possibly with a single nVidia GPU. These codes are primarily designed for a learning purpose, not for scalability. 

### Dependency 
pytorch, torchtext, tqdm 

### Experiments
#### 1. Text classification with a single-layer Transformer encoder (IMDB dataset)
This experiment was motivated from [this blog](http://www.peterbloem.nl/blog/transformers). 

To run the experiment,
```
python classification_experiment.py 
```
and optionally with tunable hyperparameters (the example below show defaults),
```
python classification_experiment.py --epochs 10 --batch 4 --max_len 512 --vocab_size 5000 --n_layer 1 --n_head 3 --d_embed 16 --position_embedding False
```
With this default configuration, you get test accuracy of around 0.86 after 10 epochs of training. Not bad for the relativey small model size and no reliance on any pretrained embedding or weights. Number of learnable parameters in the self attention layer (except embedding, normalization, and additional linear layers) is 3088. 

I'm writing a blog post for tutorial on this experiment. It will be available by the end of March 2020. So please stay tuned if you are interested in! 
