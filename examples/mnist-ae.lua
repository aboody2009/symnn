local qlua = false
if torch == nil then --qlua or luajit
   require 'torch'
   require 'trepl'
   qlua = true
end

local nn = require '../symnn'
local mnist = require 'mnist' -- https://github.com/andresy/mnist
require 'optim'
require 'image'
require 'xlua'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST Training')
cmd:text()
cmd:text('Options:')
cmd:option('-model', 'mlp', 'type of model to train: conv | mlp')
cmd:option('-dropout', false, 'use dropout in convnet')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-train', 'rmsprop', 'optimization method: sgd | rmsprop')
cmd:option('-learningRate', 0.01, 'learning rate at t=0')
cmd:option('-batchSize', 40, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0.999, 'weight decay (rmsprop only)')
cmd:option('-momentum', 0.9, 'momentum (SGD only)')
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
   model:add(nn.Linear(8*14*14, 100))
   model:add(nn.Sigmoid())
   model:add(nn.Linear(100, 784))
elseif opt.model == 'mlp' then
   model:add(nn.Reshape(784))
   model:add(nn.Linear(784, 100))
   model:add(nn.Sigmoid())
   model:add(nn.Linear(100, 784))
end
model:add(nn.Sigmoid())
model:add(nn.Reshape(28, 28))
model:add(nn.MSE())
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

local function train()
   print('training')
   model:training(true)
   for i = 1, trainData.size do
      local ex = trainData[i]
      local x = ex.x:double()
      local input = symtorch.Tensor(1, 28, 28):copy(x:clone())
      local target = x:clone():div(255)
      local stats = model:train(input, target)
      xlua.progress(i, trainData.size)
   end
end

local function test()
   print('testing')
   model:training(false)
   for i = 1, 5 do
      local ex = trainData[i]
      local x = symtorch.Tensor(1, 28, 28):copy(ex.x:double():clone())
      local output = model:forward(x).w
      if qlua then
         image.display(x.w:resize(28,28))
         image.display(output)
      else
         print(output)
      end
   end
end

train()
test()