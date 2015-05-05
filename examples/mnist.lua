require '../symnn'
local mnist = require 'mnist' -- https://github.com/andresy/mnist

local model = symnn.Sequential(4)
model:add(symnn.Reshape(784))
model:add(symnn.Linear(784, 100))
model:add(symnn.ReLU())
model:add(symnn.Linear(100, 10))
model:add(symnn.SoftMax())
model:add(symnn.ClassNLL())
print(model)

local trainData = mnist.traindataset()
--local testData = mnist.testdataset()

local accuracy = function(o, t)
   local min, argmin = o.w:min(1)
   if argmin:squeeze() == t then return 1
   else return 0 end
end

for i = 1, trainData.size do
   local ex = trainData[i]
   local x = symtorch.Tensor(28, 28):copy(ex.x:double())
   local target = ex.y + 1
   local stats = model:train(x, target, accuracy)
   print(i, '\t', stats.cost, '\t', stats.accuracy)
end