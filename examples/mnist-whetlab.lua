local nn = require '../symnn'
local mnist = require 'mnist' -- https://github.com/andresy/mnist
local whetlab = require 'whetlab'
require 'optim'
require 'xlua'
torch.manualSeed(1)

local cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST Whetlab Training')
cmd:text()
cmd:text('Options:')
cmd:option('-token', '', 'Whetlab access token')
local opt = cmd:parse(arg or {})

if opt.token == '' then
   error('Provide an access token to run this script.')
end

local whetlab_params = {
   nhidden = { type ='int', min = 20, max = 200 },
   lr = { type = 'float', min = 0, max = 0.2 },
   batchSize = { type = 'int', min = 4, max = 100 },
   activation = { type = 'enum', options = { 'sigmoid', 'tanh', 'relu' } }
}
local whetlab_outcome = { name = 'accuracy' }
local scientist = whetlab('MNIST symtorch/nn', 'Test 1', whetlab_params, whetlab_outcome, True, opt.token)
local job = scientist:suggest()

local trainData = mnist.traindataset()
local testData = mnist.testdataset()
local confusion = optim.ConfusionMatrix(10)

local function run(job)
   for k, v in pairs(job) do print(k, v) end

   local nonlinearity
   if job.activation == 'sigmoid' then
      nonlinearity = nn.Sigmoid()
   elseif job.activation == 'tanh' then
      nonlinearity = nn.Tanh()
   elseif job.activation == 'relu' then
      nonlinearity = nn.ReLU()
   else
      print('Invalid activation function')
      return nil
   end

   local model = nn.Sequential(job.batchSize)
   model:add(nn.Reshape(784))
   model:add(nn.Linear(784, job.nhidden))
   model:add(nonlinearity)
   model:add(nn.Linear(job.nhidden, 10))
   model:add(nn.SoftMax())
   model:add(nn.ClassNLL())
   symtorch.update.lr = job.lr
   symtorch.update.stepCache = {}

   model:training(true)
   for i = 1, trainData.size do
      local ex = trainData[i]
      local x = symtorch.Tensor(1, 28, 28):copy(ex.x:double())
      local target = ex.y + 1
      local stats = model:train(x, target)
      confusion:add(stats.output, target)
      xlua.progress(i, trainData.size)
   end
   print(confusion)
   local accuracy = confusion.totalValid * 100
   confusion:zero()

   model = nil
   symtorch.update.stepCache = {}
   collectgarbage()
   return accuracy
end

for i = 1, 20 do
   print('Instance ' .. i)
   print('==========')
   accuracy = run(job)
   scientist:update(job, accuracy)
   job = scientist:suggest()
end