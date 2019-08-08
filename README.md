# NER.jl
Implementation of Sequence Labelling Models for Named Entity Recognition.
These models are kept in `Sequence_models/` directory.
The NER API built is currently in `src/`, to use the NER API -

```julia
julia> include("src/ner.jl")
ner = NERTagger()
ner("remove_ner_label_prefix (generic function with 1 method)

julia> str = "John Doe works in Google, New York."
"John Doe works in Google, New York."

julia> collect(zip(ner(str), tokenize(str)))
9-element Array{Tuple{String,String},1}:
 ("PER", "John")
 ("PER", "Doe")
 ("O", "works")
 ("O", "in")
 ("ORG", "Google")
 ("O", ",")
 ("LOC", "New")
 ("LOC", "York")
 ("O", ".")
```

`valid/` has the notebooks recording the performance.
