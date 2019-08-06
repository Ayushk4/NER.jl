using TextAnalysis, Flux
include("../src/ner.jl")

function try_outs(ner::NERmodel, x_in, y_in)
    unique_labels = unique(ner.model.labels)
    num_labels = length(unique_labels)
    confusion_matrix = zeros(Int, (num_labels, num_labels))

    for (x_seq, y_seq) in zip(X_input_dev, Y_oh_dev)
        preds = ner(x_seq)

        for (pred, logit) in zip(preds, y_seq)
            confusion_matrix[findfirst(pred, unique_labels), findfirst(logit, unique_labels)] += 1
        end
    end

    s1 = sum(confusion_matrix, dims=2)
    s2 = sum(confusion_matrix, dims=1)
    dg = diag(confusion_matrix)

    a = sum(dg ./ s1) / 5
    b = sum(dg ./ s2) / 5

    println("Precision and recall are:", a, " ", b)
    println("F1 is:", (2 * a * b) / (a + b))
end
