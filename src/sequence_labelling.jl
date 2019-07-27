struct BiLSTM_CNN_CRF_Model{C, W, L, D, O, A}
    labels::Array{String, 1} # List of Labels
    char_idx::Dict{Words, Int64} # Dict that maps chars to indices in W_Char_Embed
    words_idx::Dict{String, Int64} # Dict that maps words to indices in W_word_Embed
    conv1::C # Convolution Layer over W_Char_Embed to give character representation
    W_Char_Embed::W # Weights for character embeddings
    W_word_Embed::W # Further trained GloVe Embeddings
    forward_lstm::L # Forward LSTM
    backward::L # Backward LSTM
    d_out::D # Dense_out
    c::O # CRF
    init_α::A
    device::Symbol # Cpu or Gpu
end

BiLSTM_CNN_CRF_Model(labels, char_idx, words_idx) =
            BiLSTM_CNN_CRF_Model(labels, char_idx, words_idx, :cpu)

function BiLSTM_CNN_CRF_Model(labels, char_idx, words_idx, device; CHAR_EMBED_DIMS=25, WORD_EMBED_DIMS=100,
                              CNN_OUTPUT_SIZE=30, CONV_PAD= (0,2)CONV_WINDOW_LENGTH = 3
                              LSTM_STATE_SIZE = 200)
    n = length(labels)
    init_α = fill(-10000, (n + 2, 1))
    init_α[n + 1] = 0

    BiLSTM_CNN_CRF_Model(labels, char_idx, words_idx, Conv((CHAR_EMBED_DIMS, CONV_WINDOW_LENGTH), 1=>CNN_OUTPUT_SIZE, pad=(0,2)),
                rand(CHAR_EMBED_DIMS, length(char_idx)), rand(WORD_EMBED_DIMS, length(words_idx),
                LSTM(CNN_OUTPUT_SIZE + WORD_EMBED_DIMS, LSTM_STATE_SIZE), LSTM(CNN_OUTPUT_SIZE + WORD_EMBED_DIMS, LSTM_STATE_SIZE),
                Dense(LSTM_STATE_SIZE * 2, DENSE_OUT_SIZE + 2), CRF(n), init_α
                )
end

function load_weights()
    
