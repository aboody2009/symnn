local nn = require '../symnn'
local mnist = require 'mnist' -- https://github.com/andresy/mnist

--symtorch.update = nn.sgd()

local model = nn.Sequential(40)
model:add(nn.Reshape(784))
model:add(nn.Linear(784, 100))
model:add(nn.Sigmoid())
model:add(nn.Linear(100, 10))
model:add(nn.SoftMax())
model:add(nn.ClassNLL())
print(model)

local trainData = mnist.traindataset()
--local testData = mnist.testdataset()

local accuracy = function(o, t)
   local min, argmax = o.w:max(1)
   argmax = argmax:squeeze()
   if argmax == t then return 1
   else return 0 end
end

for i = 1, trainData.size do
   local ex = trainData[i]
   local x = symtorch.Tensor(28, 28):copy(ex.x:double())
   local target = ex.y + 1
   local stats = model:train(x, target, accuracy)
   print(i, '\t', stats.cost, '\t', stats.accuracy)
end