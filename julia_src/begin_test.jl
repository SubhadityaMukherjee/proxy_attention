using Flux
using Flux: onehotbatch, onecold, flatten
using Flux.Losses: logitcrossentropy
using Flux.Data: DataLoader
using Flux: Chain, Dense, Conv, onehotbatch
using Parameters: @with_kw
using Statistics: mean
using CUDA
using MLDatasets: CIFAR10
using MLUtils: splitobs
using Images, Statistics, Zygote
using Zygote: @adjoint
using Metalhead: VGG

if CUDA.has_cuda()
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

function get_processed_data(args)
    x, y = CIFAR10(:train)[:]

    (train_x, train_y), (val_x, val_y) = splitobs((x, y), at=1-args.valsplit)

    train_x = float(train_x)
    train_y = onehotbatch(train_y, 0:9)
    val_x = float(val_x)
    val_y = onehotbatch(val_y, 0:9)
    
    return (train_x, train_y), (val_x, val_y)
end

function get_test_data()
    test_x, test_y = CIFAR10(:test)[:]
   
    test_x = float(test_x)
    test_y = onehotbatch(test_y, 0:9)
    
    return test_x, test_y
end

# VGG16 and VGG19 models
function vgg16()
    Chain([
        Conv((3, 3), 3 => 64, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(64),
        Conv((3, 3), 64 => 64, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(64),
        MaxPool((2,2)),
        Conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(128),
        Conv((3, 3), 128 => 128, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(128),
        MaxPool((2,2)),
        Conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(256),
        Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(256),
        Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(256),
        MaxPool((2,2)),
        Conv((3, 3), 256 => 512, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(512),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(512),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(512),
        MaxPool((2,2)),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(512),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(512),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(512),
        MaxPool((2,2)),
        flatten,
        Dense(512, 4096, relu),
        Dropout(0.5),
        Dense(4096, 4096, relu),
        Dropout(0.5),
        Dense(4096, 10)
    ])
end

@with_kw mutable struct Args
    batchsize::Int = 128
    lr::Float64 = 3e-4
    epochs::Int = 3
    valsplit::Float64 = 0.2
end

function train(; kws...)
    # Initialize the hyperparameters
    args = Args(; kws...)
	
    # Load the train, validation data 
    train_data, val_data = get_processed_data(args)
    
    train_loader = DataLoader(train_data, batchsize=args.batchsize, shuffle=true)
    val_loader = DataLoader(val_data, batchsize=args.batchsize)

    @info("Constructing Model")	
    m = vgg16() |> gpu

    loss(x, y) = logitcrossentropy(m(x), y)

    ## Training
    # Defining the optimizer
    opt = ADAM(args.lr)
    ps = Flux.params(m)

    @info("Training....")
    # Starting to train models
    for epoch in 1:args.epochs
        @info "Epoch $epoch"

        for (x, y) in train_loader
            x, y = x |> gpu, y |> gpu
            gs = Flux.gradient(() -> loss(x,y), ps)
            Flux.update!(opt, ps, gs)
        end

        validation_loss = 0f0
        for (x, y) in val_loader
            x, y = x |> gpu, y |> gpu
            validation_loss += loss(x, y)
        end
        validation_loss /= length(val_loader)
        @show validation_loss
    end

    return m
end

function test(m; kws...)
    args = Args(kws...)

    test_data = get_test_data()
    test_loader = DataLoader(test_data, batchsize=args.batchsize)

    correct, total = 0, 0
    for (x, y) in test_loader
        x, y = x |> gpu, y |> gpu
        correct += sum(onecold(cpu(m(x))) .== onecold(cpu(y)))
        total += size(y, 2)
    end
    test_accuracy = correct / total

    # Print the final accuracy
    @show test_accuracy
end

m = train()
test(m)



@adjoint relu(x) = relu(x), Δ -> ((Δ .* (Δ .> 0)) * (x .* (x .> 0)) .* Δ, -1)

# function get_backend(arch::String="vgg")
#     if arch == "vgg"
#         model = VGG19()
#     else
#         print("Other backends not supported yet")
#     end
#     return model
# end

H, W = 224, 224

function preprocess_input(x::AbstractArray)
    x ./= 127.5
    x .-= 1.
    return x
end

function load_image(path, preprocess::Bool=true, nsize = (224, 224))
    # image = RGB.(load(path))

    # image = map(img -> Images.imresize(img, nsize...), image)
    image = RGB.(load(path)) |> (x -> Images.imresize(1,x, 224, 224))
    # image = RGB.(load(path))
    # image = map(img -> permutedims(channelview(img), (3,2,1)), image)

    # image = reshape(image, (1,1, size(image)...))
    if preprocess == true
        # image = reshape(image, (1, size(image)...))
        # image = map(img -> permutedims(channelview(img), (3,2,1)), image)
        image = preprocess_input(image)
    end
    return image
end

function deprocess_image(image::AbstractArray)
    if ndims(image) > 3
        image = collect(Iterators.flatten(image))
    end
    x .-= mean(x)
    x ./= (std(x) .+ 1e-5)
    x .*= 0.1

    x .+= 0.5
    x = clamp(x, 0, 1)

    x .*= 255
    x = Int8.(clip(x, 0, 255))
    return x
end

function normalize(x::AbstractArray)
    return (x + 1e-1) / (sqrt(mean(x^2 + 1e-10)))
end

function target_category_loss(x, category_index, classes)
    return onehotbatch(category_index, 1:classes)
end

function guided_backprop(input_model, image)
    gradient(()->sum(input_model(image)), Flux.params(input_model))
end

test_im = "/media/hdd/github/improving_robotics_datasets/julia_src/deer.png"

# Gray.(load(test_im)) |> (x -> Images.imresize(1,x, 224, 224))
x = RGB.(Images.load(test_im))
x = Images.imresize(x, (64,64)...) # 224x224 depends on the model size
x = permutedims(channelview(x), (3,2,1))
# Channelview returns a view of A, splitting out (if necessary) the color channels of A into a new first dimension.

x = reshape(x, size(x)..., 1) # Add an extra dim to show we only have 1 image
x = float32.(x) # Convert to float32 instead of float64
x
# model(x)


# gradient(()->sum(input_model(image)), Flux.params(input_model))
# x = x |> gpu
# m = m |> gpu

m = m |> cpu
m(x)


gb = guided_backprop(m, test_im|>gpu)

size(m(test_im))