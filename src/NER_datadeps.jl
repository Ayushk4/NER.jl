using DataDeps

register(DataDep("NER Model Weights",
    """
    The weights for NER Sequence Labelling Model.
    """,
    "https://raw.githubusercontent.com/Ayushk4/Random_set_of_codes/49b1c509f49d5a7d8ab08f47b94fd33ad0d5b29a/weights.tar.xz",
    post_fetch_method = function(fn)
        unpack(fn)
        dir = "weights"
        innerfiles = readdir(dir)
        mv.(joinpath.(dir, innerfiles), innerfiles)
        rm(dir)
    end
))

register(DataDep("NER Model Dicts",
    """
    The character and words dict for NER Sequence Labelling Model.
    """,
    "https://raw.githubusercontent.com/Ayushk4/Random_set_of_codes/637995d8ebeb9a5ca67f77298bd8dc6054187457/model_dicts.tar.xz",
    post_fetch_method = function(fn)
        unpack(fn)
        dir = "model_dicts"
        innerfiles = readdir(dir)
        mv.(joinpath.(dir, innerfiles), innerfiles)
        rm(dir)
    end
))
