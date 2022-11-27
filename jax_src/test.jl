using FastAI, FastVision, FastMakie, Metalhead
import CairoMakie
using Metalhead

# Test lrfinder
data, blocks = load(datarecipes()["imagenette2-160"])
task = ImageClassificationSingle(blocks)
learner = tasklearner(task, data; callbacks=[ToGPU(), Metrics(accuracy)])
finderresult = lrfind(learner)

FastMakie.Makie.plot(finderresult)

# From scratch
data, blocks = load(datarecipes()["imagenette2-160"])
task = ImageClassificationSingle(blocks)
backbone = Metalhead.ResNet(18).layers[1:end-1]
learner = tasklearner(task, data; callbacks=[ToGPU(), Metrics(accuracy)])
fitonecycle!(learner, 10, 0.001)

# Transfer learning
learner = tasklearner(task, data;
    backbone=Metalhead.ResNet(34, pretrain=true).layers[1][1:end-1],
    callbacks=[ToGPU(), Metrics(accuracy)])
finetune!(learner, 1, 0.0005)

savetaskmodel("testimagenette.jld2", task, learner.model, force = true)

using Zygote

task, model = loadtaskmodel("testimagenette.jld2")
model = gpu(model);

samples = [getobs(data, i) for i in rand(1:numobs(data), 9)]
images = [sample[1] for sample in samples]
labels = [sample[2] for sample in samples]

preds = predictbatch(task, model, images; device = gpu, context = Validation())

# h_out = model[1:2](images)
pms = Flux.params(model[end-1:end]);

gradient(a -> )
