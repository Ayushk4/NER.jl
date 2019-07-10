using CorpusLoaders
using MultiResolutionIterators
using TextAnalysis
using TextAnalysis: CRF, crf_loss
using WordTokenizers
using Embeddings
using Flux
using Flux: onehot, param, onehotbatch, Conv, LSTM, flip, Momentum

CHAR_EMBED_DIMS = 25
WORD_EMBED_DIMS = 50
DESIRED_TAG_SCHEME = "BIOES"

train = load(CoNLL(), "train") # training set
# test = load(CoNLL(), "test") # test set
# dev = load(CoNLL(), "dev") # dev set

dataset = flatten_levels(train, lvls(CoNLL, :document)) |> full_consolidate

typeof(dataset)

X_train = [CorpusLoaders.word.(sent) for sent in dataset]
Y_train = [CorpusLoaders.named_entity.(sent) for sent in dataset]
tag_scheme!.(Y_train, "BIO2", DESIRED_TAG_SCHEME)

@assert length.(X_train) == length.(Y_train)

words_vocab = unique(vcat(X_train...))
alphabets = unique(vcat(collect.(words_vocab)...))
labels = unique(vcat(Y_train...))
DENSE_OUT_SIZE = num_labels = length(labels)

# TODO: Preprocessing: Change n't => not and Shuffle and all numbers to 0



#### Word Embeddings
# using these indices to generate the flux one hot input word embeddings vectors.

# TODO: EMBEDDINGS OF the input size.
embtable = load_embeddings(GloVe)
get_word_index = Dict(word => ii for (ii, word) in enumerate(embtable.vocab))
get_word_from_index = Dict(value => key for (key, value) in get_word_index)
W_word_Embed = embtable.embeddings

# One Vec for unknown chars
UNK_Word = "<UNK>"
UNK_Word_Idx = length(get_word_index) + 1
embedding_vocab_length = length(get_word_index) + 1
@assert UNK_Word ∉ collect(keys(get_word_index))

get_word_index[UNK_Word] = UNK_Word_Idx
get_word_from_index[UNK_Word_Idx] = UNK_Word
W_word_Embed = hcat(W_word_Embed, rand(WORD_EMBED_DIMS))
@assert size(W_word_Embed, 2) == embedding_vocab_length



##### Char Embeddings
UNK_char = '¿'
@assert UNK_char ∉ alphabets
push!(alphabets, UNK_char)
sort!(unique!(alphabets))

# using these indices to generate the flux one hot input Char embeddings vectors.

W_Char_Embed = rand(CHAR_EMBED_DIMS, length(alphabets)) .- 0.5 # Bringing into range b/w -0.5 and 0.5
W_Char_Embed = param(W_Char_Embed)
get_char_index = Dict(char => ii for (ii, char) in enumerate(alphabets))
get_char_from_index = Dict(value => key for (key, value) in get_char_index)


############# Creating onehoot vectors (useful for embeddings lookup), convenient
# TODO: Minibatches
onehotword(word) = onehot(get(get_word_index, lowercase(word), get_word_index[UNK_Word]), 1:embedding_vocab_length)
onehotchars(word) = onehotbatch([get(get_char_index, c, get_char_index[UNK_char]) for c in word], 1:length(alphabets))
onehotlabel(label) = onehot(label, labels)

oh_seq(arr, f) = [f(element) for element in arr]

X_words_train = [oh_seq(sentence, onehotword) for sentence in X_train] # A Bunch of Sequences of words, i.e. sentences

X_chars_train = [oh_seq(sentence, onehotchars) for sentence in X_train] # A Bunch Sequences of Array of Chars, done to prevent repeated computations.

Y_oh_train = [oh_seq(tags_sequence, onehotlabel) for tags_sequence in Y_train]

@assert length.(X_train) == length.(X_words_train) ==
        length.(X_chars_train) == length.(Y_oh_train)

#################### MODEL ######################
# 1. Character Embeddings with CNNs
# 2. Obtain Word Embeddings, concatenate Char & Word Embeddings
# 3. Bi-LSTM
# 4. Softmax or CRF on each unit in sequence

CNN_OUTPUT_SIZE = 53
CONV_WINDOW_LENGTH = 3
LSTM_STATE_SIZE = 253
DROPOUT_RATE_CNN = DROPOUT_INPUT_EMBEDDING = DROPOUT_OUT_LAYER = 0.68

# 1. Dropout before conv, Max poll layer, PADDING on both sides, hyperparams = window size, output vector size.
char_features = Chain(x -> W_Char_Embed * x,
                      x -> reshape(x, size(x)..., 1,1),
                      Dropout(DROPOUT_RATE_CNN),
                      Conv((CHAR_EMBED_DIMS, CONV_WINDOW_LENGTH), 1=>CNN_OUTPUT_SIZE, pad=(0,2)),
                      x -> maximum(x, dims=2),
                      x -> reshape(x, length(x),1)
                )

# 2. Input embeddings:
# Maybe could use only the embeddings for the words needed
# W_word_Embed = param(W_word_Embed) # For trainable

get_word_embedding(w) = W_word_Embed * w # works coz - onehot

# Dropout before LSTM
dropout_embed = Dropout(DROPOUT_INPUT_EMBEDDING)
input_embeddings((w, cs)) = dropout_embed(vcat(get_word_embedding(w), char_features(cs)))

# 3. Bi-LSTM

forward_lstm = LSTM(CNN_OUTPUT_SIZE + WORD_EMBED_DIMS, LSTM_STATE_SIZE)
backward_lstm(x) = flip(LSTM(CNN_OUTPUT_SIZE + WORD_EMBED_DIMS, LSTM_STATE_SIZE), x)

bilstm_layer(x) = vcat.(forward_lstm.(x), backward_lstm(x))

# TODO: Bias vectors are initialized to zero, except the bias bf for the forget gate in LSTM , which is initialized to 1.0

# 4. Softmax / CRF Layer
# Dropout after LSTM

dropout = Dropout(DROPOUT_OUT_LAYER)
m(w_cs) = dropout.(bilstm_layer(input_embeddings.(w_cs)))


using TextAnalysis: crf_loss, CRF
Flux.@treelike TextAnalysis.CRF
c = TextAnalysis.CRF(num_labels, LSTM_STATE_SIZE * 2)

loss(w_cs, y) =  crf_loss(c, m(w_cs), y)

η = 0.01 #
β = 0.05 # rate decay
ρ = 0.9 # momentum

opt = Momentum(η, ρ)

Flux.train!(loss, params(m, c)
# TODO: gradient clipping and rate decay
# batch size = 10


# function test_raw_sentence # Convert unknown to UNK and to lowercase
# TODO: Try with and without lowercased chars in char embedding
# TODO: Try with and without trainable embeddings.
