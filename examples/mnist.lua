local nn = require '../symnn'
local mnist = require 'mnist' -- https://github.com/andresy/mnist
require 'optim'
require 'xlua'

--symtorch.update = nn.sgd()
--symtorch.update.clip = 100

local model = nn.Sequential(40)
model:add(nn.Reshape(784))

-- TODO: conv

-- MLP
model:add(nn.Linear(784, 100))
model:add(nn.Sigmoid())
model:add(nn.Linear(100, 10))

-- Linear regression
--model:add(nn.Linear(784, 10))

model:add(nn.SoftMax())
model:add(nn.ClassNLL())
print(model, symtorch.update)

local trainData = mnist.traindataset()
local testData = mnist.testdataset()
local confusion = optim.ConfusionMatrix(10)

local function train()
   print('training')
   model:training(true)
   for i = 1, trainData.size do
      local ex = trainData[i]
      local x = symtorch.Tensor(28, 28):copy(ex.x:double())
      local target = ex.y + 1
      local stats = model:train(x, target)
      confusion:add(stats.output, target)
      xlua.progress(i, trainData.size)
   end
   print(confusion)
   confusion:zero()
end

local function test()
   print('testing')
   model:training(false)
   for i = 1, testData.size do
      local ex = testData[i]
      local x = symtorch.Tensor(28, 28):copy(ex.x:double())
      local target = ex.y + 1
      local output = model:forward(x).w
      confusion:add(output, target)
      xlua.progress(i, testData.size)
   end
   print(confusion)
   confusion:zero()
end

while true do
   train()
   test()
end