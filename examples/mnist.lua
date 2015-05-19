local nn = require '../symnn'
local mnist = require 'mnist' -- https://github.com/andresy/mnist
require 'optim'
require 'xlua'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST Training')
cmd:text()
cmd:text('Options:')
cmd:option('-model', 'mlp', 'type of model to train: conv | mlp | linear')
cmd:option('-dropout', false, 'use dropout in convnet')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-train', 'rmsprop', 'optimization method: sgd | rmsprop')
cmd:option('-learningRate', 0.01, 'learning rate at t=0')
cmd:option('-batchSize', 40, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0.999, 'weight decay (rmsprop only)')
cmd:option('-momentum', 0.9, 'momentum (SGD only)')
cmd:option('-epochs', 5, 'number of epochs')
cmd:option('-threads', 2, 'nb of threads to use')
cmd:text()
local opt = cmd:parse(arg or {})
print(opt)

torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)

local model = nn.Sequential(opt.batchSize)
if opt.model == 'conv' then
   model:add(nn.SpatialConvolution(8, 5, 5, 1, 2))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(2, 2, 2))
   if opt.dropout then model:add(nn.Dropout()) end
   model:add(nn.Reshape(8*14*14))
   model:add(nn.Linear(8*14*14, 10))
elseif opt.model == 'mlp' then
   model:add(nn.Reshape(784))
   model:add(nn.Linear(784, 100))
   model:add(nn.Sigmoid())
   model:add(nn.Linear(100, 10))
elseif opt.model == 'linear' then
   model:add(nn.Reshape(784))
   model:add(nn.Linear(784, 10))
end
model:add(nn.SoftMax())
model:add(nn.ClassNLL())
print(model)

if opt.train == 'sgd' then
   symtorch.update = nn.sgd {
      lr = opt.learningRate,
      momentum = opt.momentum
   }
else
   symtorch.update.lr = opt.learningRate
   symtorch.update.decayRate = opt.weightDecay
end
print(symtorch.update)

local trainData = mnist.traindataset()
local testData = mnist.testdataset()
local confusion = optim.ConfusionMatrix(10)

local accuracy = function(o, t)
   local max, argmax = o:max(1)
   argmax = argmax:squeeze()
   if argmax == t then return 1
   else return 0 end
end

local function train()
   print('training')
   model:training(true)
   for i = 1, trainData.size do
      local ex = trainData[i]
      local x = symtorch.Tensor(1, 28, 28):copy(ex.x:double())
      local target = ex.y + 1
      local stats = model:train(x, target, accuracy)
      confusion:add(stats.output, target)
      xlua.progress(i, trainData.size)
      --print(i, stats.accuracy)
   end
   print(confusion)
   confusion:zero()
end

local function test()
   print('testing')
   model:training(false)
   for i = 1, testData.size do
      local ex = testData[i]
      local x = symtorch.Tensor(1, 28, 28):copy(ex.x:double())
      local target = ex.y + 1
      local output = model:forward(x).w
      confusion:add(output, target)
      xlua.progress(i, testData.size)
   end
   print(confusion)
   confusion:zero()
end

for epoch = 1, opt.epochs do
   print('Epoch #' .. tostring(epoch))
   train()
   test()
end