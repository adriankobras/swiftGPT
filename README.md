
# swiftGPT

![nanoGPT](assets/swiftGPT.png)

This repository is a fork of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) trained on Taylor Swift lyrics. It's purpose is to get myself familiar with the building blocks of GPTs.

## overview

|model|swiftGPT-stoi|swiftGPT-bpe|<span style="white-space:nowrap;">swiftGPT-2</span>|
|----|----|----|----|
|characteristics|character-level nanoGPT|nanoGPT using the OpenAI BPE tokenizer|GPT-2 finetuned
|number of parameters|10.65M|tbd|tbd|
|layers|6|
|heads|6|
|embedding size|384|
|tokenizer|string to index|
|training data||txt file containing all of TS lyrics| |

## install

```
pip install torch numpy tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org)
- [numpy](https://numpy.org/install/)
-  `tiktoken` for OpenAI's fast BPE code
-  `wandb` for optional logging
-  `tqdm` for progress bars

## quick start

Training a character-level GPT on the works of Taylor Swift. First, we download all of Taylor Swift's lyrics in a single (0.3MB) file and turn it from raw text into one large stream of integers:

```
python data/shakespeare_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Now it is time to train the GPT. The size of it very much depends on the computational resources of your system:

**I have a Mac**: 
The default GPT has a context size of up to 256 characters, 384 feature channels, and it is a 6-layer Transformer with 6 heads in each layer. On a M1 MacBook Pro the training takes about 10 min and the best validation loss is 1.5685. For longer training times the validation loss increases while the training loss decreases, indicating overfitting.

```
python train.py config/train_shakespeare_char.py --device=mps --compile=False --max_iters=500
```

A simpler train run could look as follows, taking around 25 sec.

```
python train.py config/train_shakespeare_char.py --device=mps --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=500 --lr_decay_iters=2000 --dropout=0.0
```

To gernerate an output, use:

```
python sample.py --out_dir=out-shakespeare-char --device=mps
```

The model generates samples like this:

```
And I don't fen heart a hagn stroe
I bet you want to go

'Cause I know it care yoeher here

[Chorus]
It's just just wasn't know what yoe wasn't mething
We don't know we grow up uperfactly writt of twing?
I see back togs plane somethink to fly, start sneak what it to secome back to me?

And me come back to New Yorh YorKeep to me deep crawere like does
And I can got a back to time all new
Let you spicting to December throom and tearler

Now I could be tell be ol yoah
Welcome bright wearing so to N
```

Not bad for a character-level model after 3 minutes of training on a home computer.
Note: In case you don't use a Apple Silicon Mac, you have to set `--device=cpu`.


**I have a GPU**: 
In case you have a GPU, you can simply train the model with:

```
$ python train.py config/train_shakespeare_char.py
```

And generate outputs by:

```
$ python sample.py --out_dir=out-shakespeare-char
```

## todos

- Try different techniques to prevent overfitting 
- Use OpenAI BPE tokenizer instead of string to index
- Finetune a pretrained GPT-2 model to TS lyrics

##

Note: The original Readme can be seen [here](https://github.com/karpathy/nanoGPT/blob/master/README.md)