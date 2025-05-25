# hugging-face-course


chapter 1:
pip install "transformer[sentencepiece]"
sentencepiece is tokenizer by google, the above command makes sure that
trasformer is intsalled with sentencepiece dependency.
sentencepiece uses byte code byte pair encoding to make sure
frequently ocuured characters form a token  together.

causal modeling is when we predict next token from previous and present tokens (Causal language modeling (CLM))
mask prediction is between one (Masked language modeling (MLM))


DistilBERT is a bit of an outlier, it has less params and higher accuracy.

in decoder, the attention layer uses all the words in the sentence.

but in decoder layer, attention is only applied to the previouse words in decoder layer.
i think because of the nature of the layers in both? encoding for feature representation and decoder for generation?

the decoder block has access to all last translated tokens (the first multi-head attentiion layer), but it has also access to all the inputs from encoder layer (its second multi-head attention layer) because it needs the full context of the input to translate any next word

atteton mask is used to avoid attention for some words. 
we use padding tokens during trauniung to make all batch of similar example size for exampole,
in this case we may use attention masking.


BART and T5 are encoder-decoder
BERT is encoder only
GPT is decoder only


In GPT2 training, they used byte pair encoding technique (i think makes a token out of commonly occured tokens) to tokenize the text. then also compute positiional embeddings vectors and both the token and postional vec embneedinfs are concatenated.
The pass through masked self attention to mask the futuire tokens (make theeir attention value 0)
and some other layer, mlultiple such blocks are used.

the above setup give us hidden states are then converted to logits using some language head (like a classification head)

The language head or task head performs linear transformation to the hidden states to give us logits. Then next token in the sequence is 

# Logits (raw scores)
logits = language_head(hidden_states)  # Shape: [batch_size, seq_len, vocab_size]

# Convert to probabilities
probabilities = torch.softmax(logits, dim=-1)

# Projects hidden_size to vocabulary_size
language_head = nn.Linear(hidden_size, vocab_size) # language head is a linear layer typically
logits = language_head(hidden_states[:, -1, :])  # Last token's hidden state

so the flow is like this

embedding vectors for each token, hidden states (which is also a vectot) for each token in sequence, language head isn applied to each hidden state vector to create logits for each token
then these logits are right shifted by 1 so that we align lables with with -- current token logit pairs with next token. so right shift is required.

then  softmax is applied to transfor the logits into probabalaities an then on some criteria, toekn is selected based on the prob distri. cross entropy loss is calculated after softmax

Decoder → hidden states hᵢ

LM head (linear) → logitsᵢ

Shift logits so logitsᵢ↔labelᵢ

Softmax(logitsᵢ) → pᵢ

Cross‐entropy(pᵢ, labelᵢ) → lossᵢ

Average over i → final loss


For classificaton like tasksm, we need to undertsand richer features in text and this bi-directioonal (thus attention is applied to both sides)

BERT uses WordPiece tokenization (unklike byte pair encoding as in gpt2)
there are some special tokens as well. [CLS] to identify start of the sentence
[SEP] is used to differeantiate between a single sentenve and multiple?

If you have more than two chunks (say question, context, extra context), you need a [SEP] after each chunk. But you only have one [CLS] at the very front. You can’t reuse it mid-sequence without losing the classification summary position.

In short:

[CLS] sits at position 0 to gather everything into one vector.

[SEP] sits after each chunk so BERT knows where one piece ends and the next begins.


PLacing these tokens specefically [CLS] placement plays an important role i guess.

BERT also adds a segment embedding to denote whether a token belongs to the first or second sentence in a pair of sentences.
so like position embedding in GPT which is telling it the position within the senquence
in BERT you can dd segment embeddings which tells that this specific token belongs to first or secobd of which sentnec ina  sequence of text given

BERT is usuall pretraine with two loss functions.

one is the masking one.

the hidden state vector for the masked token is converted to logits through a linear layer
then softmax and then final output with the vocab (same as GPT)

And second training objective is the next sentence predicton. The model must predict whether sentence B follows sentence A. we can switch the sequence of the sentences and predict the position of the sentence in a sequence. 

for a sentence we want the position for, we can attached a binary classier head to predict if the
sentence is next or not  softmax over the two classes (IsNext and NotNext).

Lets say we want to classify sentences, we will take base bert model. we will convert the sentnece into tokens using WordPiece tokenizer. -- get hidden states from the base bert.
pass hidden states through feed forward or lianear lyaer to get logit vector, we get the logits for taget class as well and then compute the crooss entropy loss

Libraries (PyTorch, TensorFlow) give you a single “cross‐entropy” function that takes raw logits and internally does log_softmax + nll_loss.

Computing softmax yourself then feeding into a separate loss is slower and less numerically stable.

When you train, you don’t actually need to compute the softmax yourself—here’s why:

1. **Cross‐entropy loss combines softmax + log**

   * Libraries (PyTorch, TensorFlow) give you a single “cross‐entropy” function that takes raw logits and internally does `log_softmax` + `nll_loss`.
   * Computing softmax yourself then feeding into a separate loss is slower and less numerically stable.

2. **Argmax is the same**

   * If you just want the top class, you don’t eed actual probanbilities—whichever logit is largest is also the most probable.
   * e.g. logits `[2.0, 1.0, 0.1]` → softmax ≈ `[0.66, 0.24, 0.10]`. Both have index 0 as the max.

3. **When you do need probs**

   * At inference time, if you want confidence scores or to threshold, you **do** run `softmax(logits)` to get real probabilities.
   * But for training and deciding “which class is predicted,” skipping softmax is just faster and stable.

---

**In short:**

* **Training:** feed logits directly to `CrossEntropyLoss` (it applies softmax under the hood).
* **Inference:** optionally apply `softmax(logits)` if you need the probability values, but for picking the best class just do `argmax(logits)`.


Token classification is NER in bert
in you want to do NER, you get the hidden states, you add a token classification head
that head gives you logits for the tokens, you compare them with the logits of the actual token 
class during training

in many of these cases for tasks in BERT, you take a base bert and apply a head specific to
that task and and compute the x-entrophy loss. 

QA is span identofication. so you compute x-entrophy loss between the label span (indetify the start and the end)

BART is used for translation (as it is encoder-decoder)
input sequence is corripted and then reconsructed. Any kind of corruption technqiue can be used.
perhaps similar to mask prediction, but a sequence of masks. 
 

many models follow similar patterns despite addressing different tasks. 

In Audio wshiper tyoe models, the audio is first converted into mel spectogram.

so a featuraization step is required.

ViT and ConvNeXT can both be used for image classification; the main difference is that ViT uses an attention mechanism while ConvNeXT uses convolutions.

in ViT, large images is divded into patches and embeddings are conpuuted from patches, so imahe patches are tokens, sequence of patches is maintained using [CLS] type of tokens

Patch embeddings can be generated using traditional CNN layers.

just like text is tokenized, image can be tokenized into non-overlappable image patches.
each image patch is converted into embedding vectors.

[cls] token is always added to the start of the embedding vector of the patch.
which means [cls] token will have hidden state and will be given to
the final classification head,

oh, you need to add position embedding as well to know the position of each patch.
so cls embedding, patch embedding and path position embedding are all used and trained and
given to the encoder block.
this above all is done before giving things to encoder block. so we prepare
images in the form of embeddings in ViT before giving it to the encoder block of ViT.

so then, here is important point.

The [CLS] token is not for any specific patch, it attends to all image patchs and
aggregegates information from all the patches.

Patch tokens: Represent local features of specific image regions
[CLS] token: Represents the entire image as a global summary


[CLS] is then going to hidden state and then using a linear layer to form logits and 
then class probs through softmax.
Bottom line: [CLS] is both the design choice and the learned mechanism for global representation. You certainly could average all patches or concatenate them, but that’s heavier and almost never outperforms the standard [CLS] approach in practice.

the converstion from hidden state to logits and then classification or cross entropy loss with 
softmax is done via classification head or any head. so the head usually 
includes all this together.

Notice the parallel between ViT and BERT: both use a special token ([CLS]) to capture the overall representation, both add position information to their embeddings, and both use a Transformer encoder to process the sequence of tokens/patches.
 global aggregator they call the CLS in BERT and ViT.

 encoder models are also called auto-encoders, bidirectional.
 the encoder BERT like models are pretrained by corrupting (masking is one type of corruption) the input sequence
 and the the model learns how to reproduce the original part. like a latent space in GANS or 
 encoder-decoder models.

 The pretraining of the T5 model consists of replacing random spans of text (that can contain several words) with a single mask special token, and the task is then to predict the text that this mask token replaces.

BART AND T5Bare sequence to sequence

When in doubt about which model to use, consider:

What kind of understanding does your task need? (Bidirectional or unidirectional)
Are you generating new text or analyzing existing text?
Do you need to transform one sequence into another?
The answers to these questions will guide you toward the right architecture.



