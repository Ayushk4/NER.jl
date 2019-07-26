using CorpusLoaders
using MultiResolutionIterators
using TextAnalysis
using TextAnalysis: CRF, crf_loss
using WordTokenizers
using Embeddings
using Flux
using Flux: onehot, param, onehotbatch, Conv, LSTM, flip, Momentum, reset!, onecold
using Tracker
using BSON: @save, @load
using CuArrays
println("Dependecies Loaded")

CHAR_EMBED_DIMS = 25
WORD_EMBED_DIMS = 50
DESIRED_TAG_SCHEME = "BIOES"

train_set = load(CoNLL(), "train") # training set
# test_set = load(CoNLL(), "test") # test set
dev_set = load(CoNLL(), "dev") # dev set

dataset = flatten_levels(train_set, lvls(CoNLL, :document)) |> full_consolidate
dev_dataset = flatten_levels(dev_set, lvls(CoNLL, :document)) |> full_consolidate

X_train = [CorpusLoaders.word.(sent) for sent in dataset]
Y_train = [CorpusLoaders.named_entity.(sent) for sent in dataset]
tag_scheme!.(Y_train, "BIO2", DESIRED_TAG_SCHEME)
@assert length.(X_train) == length.(Y_train)

X_dev = [CorpusLoaders.word.(sent) for sent in dev_dataset]
Y_dev = [CorpusLoaders.named_entity.(sent) for sent in dev_dataset]
tag_scheme!.(Y_dev, "BIO2", DESIRED_TAG_SCHEME)
@assert length.(X_dev) == length.(Y_dev)

words_vocab = unique(vcat(X_train...))
alphabets = unique(vcat(collect.(words_vocab)...))
labels = unique(vcat(Y_train...))
DENSE_OUT_SIZE = num_labels = length(labels)


#### Word Embeddings
# using these indices to generate the flux one hot input word embeddings vectors.

embtable = load_embeddings(GloVe)
get_word_index = Dict(word => ii for (ii, word) in enumerate(embtable.vocab))
get_word_from_index = Dict(value => key for (key, value) in get_word_index)
W_word_Embed = (embtable.embeddings)

# One Vec for unknown chars
UNK_Word = "<UNK>"
UNK_Word_Idx = length(get_word_index) + 1
embedding_vocab_length = length(get_word_index) + 1
@assert UNK_Word ∉ collect(keys(get_word_index))

get_word_index[UNK_Word] = UNK_Word_Idx
get_word_from_index[UNK_Word_Idx] = UNK_Word
W_word_Embed = hcat(W_word_Embed, rand(WORD_EMBED_DIMS))
W_word_Embed = Float32.(W_word_Embed)
@assert size(W_word_Embed, 2) == embedding_vocab_length


##### Char Embeddings
UNK_char = '¿'
@assert UNK_char ∉ alphabets
push!(alphabets, UNK_char)
sort!(unique!(alphabets))

# using these indices to generate the flux one hot input Char embeddings vectors.

W_Char_Embed = rand(CHAR_EMBED_DIMS, length(alphabets))
W_Char_Embed = Float32.(W_Char_Embed)
W_Char_Embed = param(W_Char_Embed)
get_char_index = Dict(char => ii for (ii, char) in enumerate(alphabets))
get_char_from_index = Dict(value => key for (key, value) in get_char_index)

println("Embeddings Made")

############# Creating onehoot vectors (useful for embeddings lookup), convenient
# TODO: Minibatches
onehotword(word) = onehot(get(get_word_index, lowercase(word), get_word_index[UNK_Word]), 1:embedding_vocab_length)
onehotchars(word) = onehotbatch([get(get_char_index, c, get_char_index[UNK_char]) for c in word], 1:length(alphabets))
onehotlabel(label) = onehot(label, labels)
onehotinput(word) = (onehot(get(get_word_index, lowercase(word), get_word_index[UNK_Word]), 1:embedding_vocab_length),
                onehotbatch([get(get_char_index, c, get_char_index[UNK_char]) for c in word], 1:length(alphabets)))

oh_seq(arr, f) = [f(element) for element in arr]

# X_words_train = [oh_seq(sentence, onehotword) for sentence in X_train] # A Bunch of Sequences of words, i.e. sentences
# X_chars_train = [oh_seq(sentence, onehotchars) for sentence in X_train] # A Bunch Sequences of Array of Chars, done to prevent repeated computations.
X_input_train = [cu.(oh_seq(sentence, onehotinput)) for sentence in X_train]
Y_oh_train = [oh_seq(tags_sequence, onehotlabel) for tags_sequence in Y_train]

Y_oh_train = cu.(Y_oh_train)

X_input_dev = [oh_seq(sentence, onehotinput) for sentence in X_dev]
Y_oh_dev = [oh_seq(tags_sequence, onehotlabel) for tags_sequence in Y_dev]

@assert length.(X_train) == length.(X_input_train) == length.(Y_oh_train)

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
conv1 = Conv((CHAR_EMBED_DIMS, CONV_WINDOW_LENGTH), 1=>CNN_OUTPUT_SIZE, pad=(0,2))
dropout1 = Dropout(DROPOUT_RATE_CNN)

char_features = Chain(x -> W_Char_Embed * x,
                      x -> reshape(x, size(x)..., 1,1),
                      dropout1,
                      conv1,
                      x -> maximum(x, dims=2),
                      x -> reshape(x, length(x),1))

# 2. Input embeddings:
# Maybe could use only the embeddings for the words needed

W_word_Embed = param(W_word_Embed) |> gpu# For trainable
get_word_embedding(w) = W_word_Embed * w |> gpu# works coz - onehot

# Dropout before LSTM
dropout_embed = Dropout(DROPOUT_INPUT_EMBEDDING) |> gpu
input_embeddings((w, cs)) = dropout_embed(vcat(get_word_embedding(w), char_features(cs))) |> gpu

# 3. Bi-LSTM

forward_lstm = LSTM(CNN_OUTPUT_SIZE + WORD_EMBED_DIMS, LSTM_STATE_SIZE) |> gpu
backward = LSTM(CNN_OUTPUT_SIZE + WORD_EMBED_DIMS, LSTM_STATE_SIZE) |> gpu
backward_lstm(x) = reverse(backward.(reverse(x)))

bilstm_layer(x) = vcat.(forward_lstm.(x), backward_lstm(x))

# TODO: Bias vectors are initialized to zero, except the bias bf for the forget gate in LSTM , which is initialized to 1.0

# 4. Softmax / CRF Layer
# Dropout after LSTM

dropout2 = Dropout(DROPOUT_OUT_LAYER) |> gpu
d_out = Dense(LSTM_STATE_SIZE * 2, DENSE_OUT_SIZE + 2) |> gpu

m = Chain(x -> input_embeddings.(x),
                bilstm_layer,
                x -> dropout.(x),
                x -> d_out.(x)) |> gpu

# dropout.(bilstm_layer(input_embeddings.(w_cs))) |> gpu

using TextAnalysis: crf_loss, CRF

Flux.@treelike TextAnalysis.CRF
c = TextAnalysis.CRF(num_labels) |> gpu

init_α = fill(-10000, (c.n + 2, 1))
init_α[c.n + 1] = 0
init_α = cu(init_α)

loss(x, y) =  crf_loss(c, m(x), y, init_α)

η = 0.005 # learning rate
β = 0.05 # rate decay
ρ = 0.9 # momentum

# TODO: rate decay
opt = Flux.Optimiser(ExpDecay(β), Momentum(η, ρ))
data = zip(X_input_train, Y_oh_train)

println("Model Built")

function load_weights(conv1, W_word_Embed, W_Char_Embed, forward_lstm, backward, c, d_out)
    conv_cpu = conv1 |> cpu
    W_word_cpu = W_word_Embed |> cpu
    W_char_cpu = W_Char_Embed |> cpu
    forward_lstm_cpu = forward_lstm |> cpu
    backward_lstm_cpu = backward |> cpu
    crf_cpu = c |> cpu
    d_cpu = d_out |> cpu

    @load "./weights/conv_cpu.bson" conv_cpu
    @load "./weights/W_word_cpu.bson" W_word_cpu
    @load "./weights/W_char_cpu.bson" W_char_cpu
    @load "./weights/forward_lstm.bson" forward_lstm_cpu
    @load "./weights/backward_lstm.bson" backward_lstm_cpu
    @load "./weights/d_cpu.bson" d_cpu
    @load "./weights/crf.bson" crf_cpu

    conv1 = conv_cpu |> gpu
    W_word_Embed = W_word_cpu |> gpu
    W_Char_Embed = W_char_cpu |> gpu
    forward_lstm = forward_lstm_cpu |> gpu
    backward = backward_lstm_cpu  |> gpu
    c = crf_cpu |> gpu
    d_out = d_cpu |> gpu

    return conv1, W_word_Embed, W_Char_Embed, forward_lstm, backward, c, d_out
end

#### To load
# println(d_out.W[1])
# conv1, W_word_Embed, W_Char_Embed, forward_lstm, backward, c, d_out =
        # load_weights(conv1, W_word_Embed, W_Char_Embed, forward_lstm, backward, c, d_out)

# println(d_out.W[1])
# println("Loaded Weights")

on(tags) = [onecold(i, labels) for i in tags]

function sent_to_label(sent)
    reset!(forward_lstm)
    reset!(backward)
    d = cu.(oh_seq(tokenize(sent), onehotinput))
    x_seq = m(d)
    ohs = TextAnalysis.viterbi_decode(cpu(c), cpu.(x_seq), cpu(init_α))

    println(on(ohs))
    on(ohs)
end

compare_label_to_oh(ohs, y) = sum([(a.ix == b.ix ? 1 : 0) for (a, b) in zip(ohx, y)])
function try_outs()
    Flux.testmode!(dropout1)
    Flux.testmode!(dropout2)
    Flux.testmode!(dropout_embed)
    sent_to_label("Avik Sengupta is mentoring this.")
    sent_to_label("Avik Sengupta and oxinabox are mentoring.")
    sent_to_label("Avik Sengupta and oxinabox are mentoring Google.")
    sent_to_label("Avik Sengupta and oxinabox are mentoring in Google.")

    num_rights = 0
    num_total = 0
    for x, y in zip(X_input_dev, Y_oh_dev)
        reset!(forward_lstm)
        reset!(backward)
        x_seq = m(x)
        ohs = TextAnalysis.viterbi_decode(cpu(c), cpu.(x_seq), cpu(init_α))

        num_rights += compare_label_to_oh(ohs, y)
        num_total += length(x)
    end
    reset!(forward_lstm)
    reset!(backward)
    Flux.testmode!(dropout1, false)
    Flux.testmode!(dropout2, false)
    Flux.testmode!(dropout_embed, false)

    accu = num_rights/num_total
    println("Dev set accuracy = $(accu)")
end

function save_weights(EPOCH, conv1, W_word_Embed, W_Char_Embed, forward_lstm, backward, d_out, c)
    conv_cpu = conv1 |> cpu
    W_word_cpu = W_word_Embed |> cpu
    W_char_cpu = W_Char_Embed |> cpu
    forward_lstm_cpu = forward_lstm |> cpu
    backward_lstm_cpu = backward |> cpu
    crf_cpu = c |> cpu
    d_cpu = d_out |> cpu

    @save "./weights/conv_cpu$(EPOCH).bson" conv_cpu
    @save "./weights/W_word_cpu$(EPOCH).bson" W_word_cpu
    @save "./weights/W_char_cpu$(EPOCH).bson" W_char_cpu
    @save "./weights/forward_lstm$(EPOCH).bson" forward_lstm_cpu
    @save "./weights/backward_lstm$(EPOCH).bson" backward_lstm_cpu
    @save "./weights/d_cpu.bson$(EPOCH)" d_cpu
    @save "./weights/crf.bson$(EPOCH)" crf_cpu
end


function train(EPOCHS)
    reset!(forward_lstm)
    reset!(backward)
    for epoch in 1:EPOCHS
        println("----------------------- EPOCH : $epoch ----------------------")
        if epoch % 2 == 1
            save_weights(epoch, conv_cpu, W_word_Embed, W_Char_Embed, forward_lstm, backward, d_out, c)
            reset!(forward_lstm)
            reset!(backward)
            try_outs()
        end

        for d in data
            reset!(forward_lstm)
            reset!(backward)
            grads = Tracker.gradient(() -> loss(d[1], d[2]), params(params(conv1)..., params(W_word_Embed)..., params(W_Char_Embed)..., params(forward_lstm, backward)..., params(d_out)..., params(c)...))
            Flux.Optimise.update!(opt,  params(params(conv1)..., params(W_word_Embed)..., params(W_Char_Embed)..., params(forward_lstm, backward)..., params(d_out)..., params(c)...), grads)
        end
        reset!(forward_lstm)
        reset!(backward)
    end
end

train(50)
reset!(forward_lstm)
reset!(backward)
save_weights(epoch, conv_cpu, W_word_Embed, W_Char_Embed, forward_lstm, backward, d_out, c)
try_outs()

# batch size = 10

# function test_raw_sentence # Convert unknown to UNK and to lowercase
# Try with and without lowercased chars in char embedding
# Try with and without trainable embeddings.
# try: Preprocessing: Change n't => not and Shuffle and all numbers to 0
