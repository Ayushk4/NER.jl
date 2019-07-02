using CorpusLoaders
using MultiResolutionIterators
using TextAnalysis
using WordTokenizers
using Embeddings
using Flux
using Flux: onehot, Conv

train = load(CoNLL(), "train") # training set
# test = load(CoNLL(), "test") # test set
# dev = load(CoNLL(), "dev") # dev set

dataset = flatten_levels(train, lvls(CoNLL, :document)) |> full_consolidate

typeof(dataset)

X_train = [CorpusLoaders.word.(sent) for sent in dataset]
Y_train = [CorpusLoaders.named_entity.(sent) for sent in dataset]

# Preprocessing: Change n't => not and Shuffle

# X = X_train[1:1000]
# tag_scheme!(tags, "BIO2". "BIOES")

words_vocab = unique(vcat(X_train...))
alphabet = [:unksort(unique(vcat(collect.(words_vocab)...)))]
typeof(alphabet)

UNK = '¿'
PAD = 'ϕ'
@assert UNK ∉ alphabet && PAD ∉ alphabet
push!(alphabet, UNK)
push!(alphabet, PAD) # PAD is required for CNNs while generating embeddings.
sort!(unique!(alphabet))

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

W_Char_Embed = rand(CHAR_EMBED_DIMS, length(alphabet)) .- 0.5 # Bringing into range b/w -0.5 and 0.5
PADDING
Chain(Dropout(DROPOUT_CNN), Conv((CHAR_EMBED_DIMS, CONV_WINDOW_LENGTH), 1=>CNN_OUTPUT_SIZE))

# function test_raw_statement # Convert unknown to UNK and to lowercase
