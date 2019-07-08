using CorpusLoaders
using MultiResolutionIterators
using TextAnalysis
using WordTokenizers
using Embeddings
using Flux
using Flux: onehot, Conv, param

train = load(CoNLL(), "train") # training set
# test = load(CoNLL(), "test") # test set
# dev = load(CoNLL(), "dev") # dev set

dataset = flatten_levels(train, lvls(CoNLL, :document)) |> full_consolidate

typeof(dataset)

X_train = [CorpusLoaders.word.(sent) for sent in dataset]
Y_train = [CorpusLoaders.named_entity.(sent) for sent in dataset]

# TODO: Preprocessing: Change n't => not and Shuffle and all numbers to 0

# X = X_train[1:1000]
# tag_scheme!(tags, "BIO2". "BIOES")

words_vocab = unique(vcat(X_train...))

UNK = '¿'
alphabet = unique(vcat(collect.(words_vocab)...))
@assert UNK ∉ alphabet
push!(alphabet, UNK)
sort!(unique!(alphabet))

onehot(, vocab) = [onehot(word, vocab) for word in sentence]

batches(xs,)

# Char Embeddings, using these indices to generate the flux one hot input word embeddings vectors.

W_Char_Embed = rand(CHAR_EMBED_DIMS, length(alphabet)) .- 0.5 # Bringing into range b/w -0.5 and 0.5
W_Char_Embed = param(W_Char_Embed)

# Word Embeddings, using these indices to generate the flux one hot input word embeddings vectors.

embtable = load_embeddings(GloVe)
get_word_index = Dict(word=>ii for (ii,word) in enumerate(embtable.vocab))
get_word_from_index = Dict(value => key for (key, value) in get_word_index)

# 1. Character Embeddings - Take care of unknown, add padding.
# 2. Word Embeddings. - Take care of unknown
# 3. hcat Character and Word Embeddings
# 4. Bi-LSTM
# 5. Softmax or CRF

CHAR_EMBED_DIMS = 25
CNN_OUTPUT_SIZE = 53

CONV_WINDOW_LENGTH = 3
LSTM_STATE_SIZE = 253
DROPOUT_CNN = 0.68

# 1. Dropout before conv, Max poll layer, PADDING on both sides, hyperparams = window size, output vector size.

char_features = Chain(x -> reshape(x, size(x)..., 1,1),
      Dropout(DROPOUT_CNN),
      Conv((CHAR_EMBED_DIMS, CONV_WINDOW_LENGTH), 1=>CNN_OUTPUT_SIZE, pad=(0,2)),
      x -> maximum(x, dims=2)
      )

# 2. Word Embeddings

# Maybe could use only the embeddings for the words needed
W_word_Embed = embtable.embeddings
# W_word_Embed = param(embtable.embeddings)

get_word_embedding(word) = W_word_Embed[:, get_word_index[word]]

# 3. Final input Embeddings

input_embeddings(w, cs) = hcat(get_word_embedding(w), dropdims(char_features(cs)))

# 4. Bi-LSTM

# TODO: Bias vectors are initialized to zero, except the bias bf for the forget gate in LSTM , which is initialized to 1.0
m(x, y) = Chain()
# function test_raw_sentence # Convert unknown to UNK and to lowercase
