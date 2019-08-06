using TextAnalysis, Flux
using LinearAlgebra
# include("../src/ner.jl")

function try_outs(ner, x_in, y_in)
    unique_labels = unique(ner.model.labels)
    num_labels = length(unique_labels)
    confusion_matrix = zeros(Int, (num_labels, num_labels))

    for (x_seq, y_seq) in zip(x_in, y_in)
        preds = ner(x_seq)

        for (pred, logit) in zip(preds, y_seq)
            confusion_matrix[findfirst(x -> x==pred, unique_labels), findfirst(x -> x==logit, unique_labels)] += 1
        end
    end

    # print(confusion_matrix)
    s1 = sum(confusion_matrix, dims=2)
    s2 = sum(confusion_matrix, dims=1)
    dg = diag(confusion_matrix)

    a = sum(dg ./ s1) / num_labels
    b = sum(dg ./ s2') / num_labels

    println("Precision and recall are:", a, " ", b)
    println("F1 is:", (2 * a * b) / (a + b))
end
