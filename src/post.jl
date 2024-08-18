struct DFImage
    subpath::String
end
Base.isless(::DFImage, ::DFImage) = false

struct DFTypst{T}
    names::Vector{String}
end


function _get_run_hparams(path::String)
    file = first(readdir(path))
    f = open(joinpath(path, file), "r")
    s = TensorBoardLogger.read_event(f)
    SESSION_START_INFO_TAG = "_hparams_/session_start_info"
  
    while s.what.name != :summary || s.what.value.value[1].tag != SESSION_START_INFO_TAG
        s = TensorBoardLogger.read_event(f)
    end
  
    hparams_metadata_encoded_bytes = s.what.value.value[1].metadata.plugin_data.content
    d = TensorBoardLogger.ProtoDecoder(IOBuffer(deepcopy(hparams_metadata_encoded_bytes)))
    decoded_content = PB.Codecs.decode(d, TensorBoardLogger.HP.HParamsPluginData)
    decoded_session_info = decoded_content.data.value.hparams
    hparams = Dict{String, Union{String, Bool, Real}}()
    for (k, v) in decoded_session_info
      push!(hparams, k => v.kind.value)
    end
    return hparams
end

function get_tb_hparams(tb_path::String)
    runs = parse_runs(tb_path)
    dfs = DataFrame()
    for run in runs 
        hp = _get_run_hparams(joinpath(tb_path, run))
        hp["run_i"] = run
        append!(dfs, DataFrame(hp))
    end

    return dfs[!, Cols(Not("run_i"), "run_i")]
end

function get_run_data(path::String; ignore_tags::Vector{Symbol}= [Symbol("host/base")], kwargs...)
    tb = TBReader(path)
    hist = MVHistory()

    TensorBoardLogger.map_summaries(tb; kwargs...) do tag, iter, val
        push!(hist, Symbol(tag), iter, val)
    end
    s = hist.storage
    return s
end

function parse_runs(tb_path::String, runs::Union{Vector{String}, Vector{Int}, Nothing}= nothing)
    if isnothing(runs)
        runs = readdir(tb_path)
    elseif runs isa Vector{Int}
        runs = map(i -> "run_$(i)", runs)
    end

    valid_runs = filter(x -> isdir(joinpath(tb_path, x)), runs)

    if length(valid_runs) != length(runs)
        @warn "some files in the subdir are wrong, please check!"
    end

    return valid_runs
end

function domain2mp4(tb_path, runs= nothing; fr::Int= 5)
    runs = parse_runs(tb_path, runs)
    
    encoder_options = (color_range=2, crf=0, preset="veryslow")
    tag = "domain/χ"

    for run in runs
        run_path = joinpath(tb_path, run)
        df_img = get_run_data(run_path; tags=tag)
        img_list = df_img[!, tag]

        file = run_path*".mp4"
        infile = run_path*"_.mp4"
        first_img = first(img_list)
        sz = map(x -> x ÷ 2 * 2, size(first_img))
        open_video_out(infile, eltype(first_img), sz, framerate=fr, encoder_options=encoder_options) do writer
            for img in img_list
                write(writer, img)
            end
        end
        cmd = `ffmpeg -i $(infile) -r 24 $(file) -y`
        readchomp(cmd)
        rm(infile)
    end
    return nothing
end

Plots.plot!(::Nothing, args...; kwargs...) = nothing
Plots.png(::Nothing, args...; kwargs...) = nothing
function post_tb_data(key, scalar_tags::Matrix{String}, data_path::String; image_tag::String = "domain/χ")
    tb_path = joinpath(data_path, "tb")
    @assert isdir(tb_path) "tb path does not exist."
    hp = get_tb_hparams(tb_path)

    post_path = joinpath(data_path, "post")
    mkpath(post_path)
    @info "all post-processed data will be stored in $post_path"

    img_path = joinpath(post_path, "images")
    typst_img_path = "images"
    mkpath(img_path)
    @info "processing images... all images will be stored in $img_path"
    
    df_results = DataFrame()
    _all = nrow(hp)
    _done = 0
    for I = 1:_all
        run_i = hp[I, :run_i]
        scalars = get_run_data(joinpath(tb_path, run_i), tags=scalar_tags)
        images = get_run_data(joinpath(tb_path, run_i), tags=image_tag)

        img_init_chi = "init_chi_$(run_i).png"
        img_trajectory = "trajectory_$(run_i).png"
        img_chis = map(1:length(scalar_tags)) do j
            return "chi_$(run_i)_tag_$(j).png"
        end

        _path_img_init_chi = joinpath(img_path, img_init_chi)
        isfile(_path_img_init_chi) || save(_path_img_init_chi, first( images[Symbol(image_tag)].values ))
        
        _path_img_trajectory = joinpath(img_path, img_trajectory)
        fig_trajectory = isfile(_path_img_trajectory) ? nothing : plot(title= "Objective Functional", xlabel= "iteration", ylabel= "value")
        
        run_i_min_vals = zeros(length(scalar_tags))
        for j = eachindex(scalar_tags)
            img_chi_path = joinpath(img_path, img_chis[j])
            tag = scalar_tags[j]
            h = scalars[Symbol(tag)]
            val, step = findmin(h.values)
            run_i_min_vals[j] = round(val; digits= 1)

            J = min(step + 10, length(h.iterations))
            plot!(fig_trajectory, h.iterations[1:J], h.values[1:J], label= tag, linewidth= 3)
            
            plot!(fig_trajectory, [step], [val], seriestype=:scatter, label= @sprintf("%.1f", run_i_min_vals[j]))
            isfile(img_chi_path) || save(img_chi_path, images[Symbol(image_tag)].values[step])
        end
        png(fig_trajectory, _path_img_trajectory)
        append!(df_results, DataFrame(
            scalar_tags[1] => run_i_min_vals[1],
            "chi_"*scalar_tags[1] => DFImage(joinpath(typst_img_path, img_chis[1])),
            scalar_tags[2] => run_i_min_vals[2],
            "chi_"*scalar_tags[2] => DFImage(joinpath(typst_img_path, img_chis[2])),
            "trajectory" => DFImage(joinpath(typst_img_path, img_trajectory)),
            "run_i" => run_i,
        ))
        _done += 1
        @info "$run_i ($_done // $_all) completed."
    end
    df_typst_hp = DFTypst{:HP}([x for x in names(hp) if (x != key && x !="run_i")])
    df_typst_data = DFTypst{:DATA}(vcat(key, names(df_results)))

    hp = rightjoin(hp, df_results; on= "run_i")
    grp_results = groupby(hp, Not(names(df_results), key))

    return grp_results, df_typst_hp, df_typst_data
end



function _parse_item(::Val{F}, x::Float64) where F 
    str = format(Format("%.1f"), x)
    if F 
        str = "#box(stroke: green, inset: 4pt)[$str]"
    end
    return str
end
_parse_item(::Val{F}, x::DFImage) where F = "#image(\"$(x.subpath)\")"
_parse_item(::Val{F}, x) where F = string(x)
function parse_to_typ(df::AbstractDataFrame, dftyp::DFTypst{:HP}, grp_id)
    dfnames = dftyp.names
    n = length(dfnames)
    title = "th"*join(["[$item]" for item in dfnames], "")
    data = "tr"*join(["[$(df[1, nm])]" for nm in dfnames ], "")
    typ_str = """
    = Case $grp_id
    == Config
    #easytable({
    let th = th.with(trans: emph)
    let tr = tr.with(
        cell_style: (x: none, y: none)
        => (fill: if calc.even(y) {
            luma(95%)
        } else {
            none
        })
    )
    cstyle(..(center + horizon,)*$n)
    cwidth(..(1fr,)*$n)
    $title
    $data
    })
    """
    return typ_str
end
function parse_to_typ(df::AbstractDataFrame, dftyp::DFTypst{:DATA})
    dfnames = dftyp.names
    n = length(dfnames)


    title = "th"*join(["[$item]" for item in dfnames], "")
    argmin_id = map(argmin, eachcol(df[!, dfnames]))
  
    data = [
       "tr"*join(["[$(_parse_item(Val(i == j), df[i, nm]))]" for (j, nm) in enumerate(dfnames)], "") for i = 1:nrow(df)
    ]
  
    data_str = """
    == Results
    #easytable({
    let th = th.with(trans: emph)
    let tr = tr.with(
        cell_style: (x: none, y: none)
        => (fill: if calc.even(y) {
            luma(95%)
        } else {
            none
        })
    )
    cstyle(..(center + horizon,)*$n)
    cwidth(..(1fr,)*$n)
    $title
    $(join(data, "\n"))
    })
    """
    return data_str
end

function my_output_typst(path)
    scalar_tags = [
        "energy/E" "energy/Jt"
    ]
    key = "scheme"

    post_path = joinpath(path, "post")
    grp_results, df_typst_hp, df_typst_data = HeatFlowTopOpt.post_tb_data(key, scalar_tags, path)

    open(joinpath(post_path, "main.typ"), "w") do io 
        print(io, """
    #import "@preview/easytable:0.1.0": easytable, elem
    #import elem: *
    """)

        for i = 1:length(grp_results)
            print(io, parse_to_typ(grp_results[i], df_typst_hp, i))
            print(io, parse_to_typ(grp_results[i], df_typst_data))
        end
        print(io, "#line(length: 100%, stroke: red + 2pt)")
        
    end

    return nothing
end

