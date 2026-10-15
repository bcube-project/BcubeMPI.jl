using Test
using SparseArrays
using ReferenceTests
using MPIUtils

function run_mpi(; nprocs = 1, filepath)
    dir = dirname(filepath)
    mpiexec() do cmd
        run(`$cmd -n $nprocs $(Base.julia_cmd()) --project=$dir $filepath`)
    end
end

"""
Custom way to "include" a file to print infos.
"""
function custom_include(path)
    filename = split(path, "/")[end]
    basename = first(splitext(filename))
    @testset "$basename" begin
        include(path)
    end
    println("done.")
end

#------------------------------------------------#
#  Functions for tests based on checksum         #
#------------------------------------------------#

"""
    scientific_format(x, digits = 10)

Return the value of `x` as a `string` in a scientific format
(decimal exponent notation). Optional argument `digits` specifies
the exact number of digits to appear after the decimal point character.
The default precision is `10`.

# Examples
```julia-repl
julia> scientific_format(-0.00001)
"-1.0000000000e-05"
julia> scientific_format(pi)
"3.1415926536e+00"
```
"""
scientific_format(x, digits = 10) = format(Format("%10.$(digits)e"), x)

"""
    get_ref_checksum(key)

Read the checksum value corresponding to the `key` in
file "checksums.sha256" in the current direction.
If `key` is not found, it returns `nothing`.
"""
function get_ref_checksum(key)
    f = readdlm(joinpath(@__DIR__, "checksums.sha256"), String)
    fname2sum = Dict(r[2] => r[1] for r in eachrow(f))
    return get(fname2sum, key, nothing)
end

"""
    compute_checksum(array::AbstractArray{<:Number}; digits = 10)

Compute the sha256 checksum of the given `array`, which be must be
a subtype of `AbstractArray{<:Number}`. The array values are rounded
and converted to string in decimal exponent notation before
computing the checksum. Optional argument `digits` specifies the exact
number of digits that are kept after the decimal point character.
The default precision is `10`.
"""
function compute_checksum(array::AbstractArray{<:Number}; digits = 10)
    path, _ = mktemp()
    str_array = map(Base.Fix2(scientific_format, digits), round.(array; digits = digits))
    writedlm(path, str_array)
    checksum = bytes2hex(open(sha256, path))
    return checksum
end

"""
    compare_checksum(key, value; digits = 10, verbose = true)

Compare the checksum of `value` with the reference checksum
corresponding to the `key` in file "checksums.sha256".
Keyword argument `digits` can be use to control rounding of `value`
before computing the checksum. This option allows to control test
failure due to floating-point precision.
Note that the reference checksum must be generated with the same number of `digits`.
"""
function compare_checksum(key, value; digits = 10, verbose = true)
    _ref = get_ref_checksum(key)
    _cur = compute_checksum(value; digits = digits)
    result = _cur == _ref
    if (!result) && verbose
        printstyled("   ➡ TEST FAIL : key = ", key, " \n"; color = :light_black)
        printstyled("      ↪ checksum ref = ", _ref, " \n"; color = :light_black)
        printstyled("      ↪ checksum     = ", _cur, " \n"; color = :light_black)
    end
    return result
end

#------------------------------------------------#
#  Functions for tests based on ReferenceTest.jl #
#------------------------------------------------#

"""
    refpath(filename::String)

Return the full path of the reference file named `filename`.
It assumes that the file is located in the `test/references/`
directory of the current project.
"""
refpath(filename::String) = joinpath(@__DIR__, "references/", filename)

"""
    compare(
        d1::Dict{String, <:AbstractArray},
        d2::Dict{String, <:AbstractArray},
        rtol,
        atol,
    )

Return `true` if dictionaries `d1` and `d2` have the
same keys and if these keys are associated with arrays
of equal values, with a relative tolerance `rtol` and
an absolute tolerance `atol`.
"""
function compare(
    d1::Dict{String, <:AbstractArray},
    d2::Dict{String, <:AbstractArray};
    atol,
    rtol,
)
    # From the official doc for `Dict`:
    # -
    # "When the values are stored internally in a hash table,
    # as is the case for Dict, the order in which they are returned may vary".
    # -
    # Then `keys(d1) == keys(d2)` is not recommended
    # and we do the following two tests instead :
    length(keys(d1)) != length(keys(d2)) && return false
    any(x -> !haskey(d1, x), keys(d2)) && return false

    for key in keys(d1)
        _x = d1[key]
        _y = d2[key]

        # We mainly rely on `isapprox` (when everything's fine)
        # and only perform additionnal checks to explain failure if needed
        check_failed = any((!isapprox).(d1[key], d2[key]; atol, rtol))

        if check_failed
            println("Comparison failure info: atol=$atol, rtol=$rtol")
            # We perform the check on all entries, without stopping on the first failure
            # (except for Inf/NaN)
            for (x, y) in zip(_x, _y)
                # Code below adapted from "isapprox"
                if !isfinite(x) || !isfinite(y)
                    println("isfinite check failed")
                    break
                end
                if isnan(x) || isnan(y)
                    println("isnan check failed")
                    break
                end

                x′, y′ = promote(x, y) # to avoid integer overflow
                n = norm(x - y)
                r = rtol * max(norm(x′), norm(y′))
                if n > max(atol, r)
                    println(
                        "x=$x, y=$y, norm(x-y)=$n, rtol*max(norm(x),norm(y))=$(rtol*max(norm(x),norm(y)))",
                    )
                end
            end
            return false
        end
    end
    return true
end
compare(; atol::Real = 1.0e-12, rtol::Real = 1.0e-12) = (a, b) -> compare(a, b; atol, rtol)

"""
Use with
```
counts = @custom_test begin
    # Some tests
    @test 1 + 1 == 2
    @test isapprox(sqrt(4), 2)
end
```
And the `counts` contains the `Test.get_test_counts` result.
"""
macro custom_test(action)
    quote
        local ts

        if VERSION ≥ v"1.13"
            ts = @testset CollectTestSet "Custom tests" begin
                $(esc(action))
            end
        else
            ts = Test.DefaultTestSet()
            Test.push_testset(ts)
            try
                $(esc(action))
            finally
                Test.pop_testset()
            end
        end
        results = Test.get_test_counts(ts)

        local counts
        if VERSION < v"1.11"
            # passes, fails, errors, broken, c_passes, c_fails, c_errors, c_broken, duration
            counts = (;
                passes = results[1],
                fails = results[2],
                errors = results[3],
                broken = results[4],
            )
        else
            counts = (;
                passes = results.passes,
                errors = results.errors,
                fails = results.fails,
                broken = results.broken,
            )
        end
        # return counts
        counts
    end
end

"""
    test_ref(filename_ref::String, data, comp::Function = compare())

Test that the values `data` with reference `filename_ref` (stored in `./test/references`)
using equality test strategy given by `comp`.

By default, `comp=compare()` assumes `data` is a subtype of
`Dict{String, <:AbstractArray}`.

If `data` is not a valid subtype for the defaut test strategy, one could :
- defined the method `_as_dict(data)` to convert `data` to a valid subtype.
- provide another test function with the keyword argument `comp`.
The function must have the same signature as `Base.isequal` function.
"""
function test_ref(filename_ref::String, data, comp::Function = compare())
    return @custom_test begin
        println("Testing against $(filename_ref)")
        @test_reference refpath(filename_ref) _as_dict(data) by = comp
    end
end

_as_dict(a::Dict) = a
_as_dict(a::AbstractArray) = Dict("array" => a)
_as_dict(a::AbstractSparseMatrix) = Dict(zip("sparse_" .* ("I", "J", "V"), findnz(a)))

""" Indicate that a given checkpoint has been reached """
function checkpoint_reached(msg)
    @testset "Reached checkpoint : $msg" begin
        @test true
    end
end

mutable struct CollectTestSet <: Test.AbstractTestSet
    description::String
    showtiming::Bool
    results::Vector{Any}

    CollectTestSet(desc; showtiming = false, kwargs...) = new(desc, showtiming, Any[])
end

Test.record(ts::CollectTestSet, t::Test.Result) = push!(ts.results, t)

Test.record(ts::CollectTestSet, t::Test.AbstractTestSet) = push!(ts.results, t)

function Test.finish(ts::CollectTestSet)
    if Test.get_testset_depth() > 0
        Test.record(Test.get_testset(), ts)
    end

    return ts
end

Test.results(ts::CollectTestSet) = ts.results

function Test.get_test_counts(ts::CollectTestSet)
    passes = 0
    fails = 0
    errors = 0
    broken = 0

    c_passes = 0
    c_fails = 0
    c_errors = 0
    c_broken = 0

    for t in ts.results
        if t isa Test.Pass
            passes += 1
        elseif t isa Test.Fail
            fails += 1
        elseif t isa Test.Error
            errors += 1
        elseif t isa Test.Broken
            broken += 1
        elseif t isa Test.AbstractTestSet
            tc = Test.get_test_counts(t)

            c_passes += tc.passes + tc.cumulative_passes
            c_fails += tc.fails + tc.cumulative_fails
            c_errors += tc.errors + tc.cumulative_errors
            c_broken += tc.broken + tc.cumulative_broken
        end
    end

    return Test.TestCounts(
        true,
        passes,
        fails,
        errors,
        broken,
        c_passes,
        c_fails,
        c_errors,
        c_broken,
        Test.format_duration(ts),
    )
end

function reduce_counts(counts, comm)
    @show counts
    sendbuf = collect(counts)
    recvbuf = similar(sendbuf)
    MPI.Reduce!(sendbuf, recvbuf, MPI.SUM, 0, comm)
    return (MPI.Comm_rank(comm) == 0) ? Tuple(recvbuf) : nothing
end

"""
Collective operation if `comm ≠ nothing` : sum tests counts and write to file
"""
function write_test_counts(filepath, counts, comm = nothing; append = false)
    counts_sum = isnothing(comm) ? counts : reduce_counts(counts, comm)

    @only_root begin
        if append
            old_counts = read_test_counts(filepath)
            for k in keys(old_counts)
                counts_sum[k] += old_counts[k]
            end
        end

        open(filepath, "w") do io
            println(io, "passes,errors,fails,broken")
            println(io, join(counts_sum, ","))
        end
    end
end

function read_test_counts(filepath)
    lines = readlines(filepath)
    line = split(lines[2], ',')
    passes, errors, fails, broken = parse.(Int, line)
    return (; passes, errors, fails, broken)
end

function print_centered_with_hash(s::String, n::Int)
    @assert n > 4 + length(s)

    total_padding = n - 4 - length(s)
    left_padding = div(total_padding, 2)
    right_padding = total_padding - left_padding

    centered = "##" * " "^left_padding * s * " "^right_padding * "##"

    println(repeat("#", n))
    println(repeat("#", n))
    println("##" * " "^(n - 4) * "##")
    println(centered)
    println("##" * " "^(n - 4) * "##")
    println(repeat("#", n))
    println(repeat("#", n))
end
