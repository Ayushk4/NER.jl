using CorpusLoaders
using MultiResolutionIterators
using TextAnalysis
using WordTokenizers
using Embeddings

train = load(CoNLL(), "train") # training set
# test = load(CoNLL(), "test") # test set
# dev = load(CoNLL(), "dev") # dev set

dataset = flatten_levels(train, lvls(CoNLL, :document)) |> full_consolidate

typeof(dataset)

X_train = [CorpusLoaders.word.(sent) for sent in dataset]
Y_train = [CorpusLoaders.named_entity.(sent) for sent in dataset]

# Preprocessing: Change n't => not

# X = X_train[1:1000]
# tag_scheme!(tags, "BIO2". "BIOES")

words_vocab = unique(vcat(X_train...))
alphabet = sort(unique(vcat(collect.(words_vocab)...)))
push!(pad and unk)

# 1. Character Embeddings - Take care of unknown, add padding.
# 2. Word Embeddings.
# 3. hcat Character and Word Embeddings
# 4. Bi-LSTM
# 5. Softmax
