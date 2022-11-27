using Flux
using Flux: onehotbatch, onecold, flatten
using Flux.Losses: logitcrossentropy
using Flux.Data: DataLoader
using Parameters: @with_kw
using Statistics: mean
using CUDA
using MLDatasets: CIFAR10
using MLUtils: splitobs
using ProgressMeter: @showprogress
using ExplainableAI
using HTTP, FileIO, ImageMagick  
using ImageShow


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

# # VGG16 and VGG19 models
function vgg16(nclasses::Int)
    Chain([
        Conv((3, 3), 3 => 64, relu, pad=(1, 1), stride=(1, 1)),
        Conv((3, 3), 64 => 64, relu, pad=(1, 1), stride=(1, 1)),
        MaxPool((2,2)),
        Conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
        Conv((3, 3), 128 => 128, relu, pad=(1, 1), stride=(1, 1)),
        MaxPool((2,2)),
        Conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
        Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
        Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
        MaxPool((2,2)),
        Conv((3, 3), 256 => 512, relu, pad=(1, 1), stride=(1, 1)),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
        MaxPool((2,2)),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
        MaxPool((2,2)),
        flatten,
        Dense(512, 4096, relu),
        Dense(4096, 4096, relu),
        Dense(4096, nclasses)
    ])
end

@with_kw mutable struct Args
    batchsize::Int = 128
    lr::Float64 = 3e-4
    epochs::Int = 3
    valsplit::Float64 = 0.1
    nclasses::Int = 10
end

function train(; kws...)
    # Initialize the hyperparameters
    args = Args(; kws...)
	
    # Load the train, validation data 
    train_data, val_data = get_processed_data(args)
    
    train_loader = DataLoader(train_data, batchsize=args.batchsize, shuffle=true)
    val_loader = DataLoader(val_data, batchsize=args.batchsize)

    @info("Constructing Model")	
    m = vgg16(args.nclasses) |> gpu

    loss(x, y) = logitcrossentropy(m(x), y)

    ## Training
    # Defining the optimizer
    opt = ADAM(args.lr)
    ps = Flux.params(m)

    @info("Training....")
    # Starting to train models
    for epoch in 1:args.epochs
        @info "Epoch $epoch"

        @showprogress for (x, y) in train_loader
            x, y = x |> gpu, y |> gpu
            gs = Flux.gradient(() -> loss(x,y), ps)
            Flux.update!(opt, ps, gs)
        end

        validation_loss = 0f0
        @showprogress for (x, y) in val_loader
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
    @showprogress for (x, y) in test_loader
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

# XAI Bit
convert_model(m) = strip_softmax(flatten_chain(m)) |> cpu
load_and_preprocess_imagenet(image) = reshape(preprocess_imagenet(load(image)), 224, 224, 3, :)

function extend_dims(A,which_dim)
       s = [size(A)...]
       insert!(s,which_dim,1)
       return reshape(A, s...)
       end

cd("/media/hdd/github/improving_robotics_datasets/jax_src/")
ims_all = readdir("./data_testing/",join= true )

ims = load_and_preprocess_imagenet.(ims_all)

model = convert_model(m[1:end-5])
analyzer = LRP(model)
# expl = heatmap(input, analyzer,)
heat = heatmap(ims[2], analyzer);
heat
test_im = load_and_preprocess_imagenet(ims_all[2])

# Add proxy attention here. Not multiply but convert in image based on value.

im_converted = test_im .* heat

# im_converted = permutedims(im_converted, (3, 1, 2))

# im_converted = extend_dims(im_converted, 1)

im_converted = colorview(RGB, im_converted/ 255)
im_converted = map(clamp01nan, im_converted)


# im_converted = extend_dims(im_converted, 1)

# im_converted = colorview(RGB, im_converted);

save("est.png",im_converted[:,:,:]);

# im_converted = extend_dims(im_converted, 4)


# im_converted[:,:,:]