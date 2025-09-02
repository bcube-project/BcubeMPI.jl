push!(LOAD_PATH, "../src/")

using BcubeParallel
using Documenter
using Literate

# Alias for `Literate.markdown`
function gen_markdown(src, name, dir)
    Literate.markdown(joinpath(src, name), dir; documenter = false, execute = false)
end

# Generate tutorials
# `documenter = false` to avoid Documenter to execute cells
tutorial_src = joinpath(@__DIR__, "..", "tutorial")
tutorial_dir = joinpath(@__DIR__, "src", "tutorial")
Sys.rm(tutorial_dir; recursive = true, force = true)
gen_markdown(tutorial_src, "helmholtz.jl", tutorial_dir)
#gen_markdown(tutorial_src, "linear_transport.jl", tutorial_dir)
#gen_markdown(tutorial_src, "flat_heater.jl", tutorial_dir)

makedocs(;
    modules = [BcubeParallel],
    authors = "ghislainb, lokmanb and bmxam",
    sitename = "BcubeParallel",
    clean = true,
    doctest = false,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://ghislainb.github.io/bcubeparallel",
        assets = String[],
    ),
    pages = ["Home" => "index.md", "Tutorials" => Any["tutorial/helmholtz.md",
    #"tutorial/linear_transport.md",
    #"tutorial/flat_heater.md",
    ]],
)

deploydocs(; repo = "github.com/ghislainb/BcubeParallel.git", push_preview = true)
