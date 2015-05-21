if torch == nil then --luajit
   require 'torch'
   require 'trepl'
end

local nn = require '../symnn'
local tds = require 'tds' -- https://github.com/torch/tds
require 'xlua'
require './utils'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('RNNLM Training')
cmd:text()
cmd:text('Options:')
cmd:option('-reclayer', 'lstm', 'rnn | lstm')
cmd:option('-dropout', true, 'use dropout in convnet')
cmd:option('-train', 'sgd', 'optimization method: sgd | rmsprop')
cmd:option('-learningRate', 0.7, 'learning rate at t=0')
cmd:option('-batchSize', 10, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0.999, 'weight decay (rmsprop only)')
cmd:option('-momentum', 0.0, 'momentum (SGD only)')
cmd:option('-epochs', 5, 'number of epochs')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'nb of threads to use')
cmd:option('-debug', false, 'set to true for lots of intermediary printing')
cmd:option('-trainSize', 'tiny', 'tiny (1/10th data) | small (1/2 dath) | full (all data)')
cmd:text()
local opt = cmd:parse(arg or {})
print(opt)

torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)

local trainFactor
if opt.trainSize == 'tiny' then
   trainFactor = 0.1
elseif opt.trainFactor == 'small' then
   trainFactor = 0.5
else
   trainFactor = 1
end

vocab = tds.hash()
vocab.word2idx = tds.hash()
vocab.idx2word = tds.hash()
vocab.size = 0

trainData = tds.hash()
trainData.data = tds.hash()
trainData.size = 0

testData = tds.hash()
testData.data = tds.hash()
testData.size = 0

model = nn.Sequential(opt.batchSize)
prev = nil
training = true

local function make_vocab()
   local continue = true
   local idx = 1
   local file = torch.DiskFile('data/ptb/train')
   while continue do
      try {
         function()
            local s = file:readString('*l'):sub(2)
            for word in s:gsplit(' ') do
               if vocab.word2idx[word] == nil then
                  vocab.word2idx[word] = idx
                  vocab.idx2word[idx] = word
                  idx = idx + 1
               end
            end
         end,
         catch = function(err)
            continue = false
         end
      }
   end

   file:close()
   vocab.size = #vocab.idx2word
   print('Vocab of size ' .. vocab.size)
end

local function make_data(train)
   local dataset, fname
   local x, target

   if train then
      dataset = trainData
      fname = 'train'
   else
      dataset = testData
      fname = 'test'
   end

   local file = torch.DiskFile('data/ptb/' .. fname)
   local idx = 1
   local continue = true
   while continue do
      try {
         function()
            local s = file:readString('*l'):sub(2)
            for word in s:gsplit(' ') do
               if x then
                  target = vocab.word2idx[word]
                  dataset.data[idx] = tds.hash()
                  dataset.data[idx].x = x
                  dataset.data[idx].y = target
                  idx = idx + 1
               end
               x = vocab.word2idx[word]
            end
         end,
         catch = function(err)
            continue = false
         end
      }
   end

   file:close()
   if train then
      dataset.size = math.floor(idx * trainFactor)
   else
      dataset.size = idx
   end
   print(fname .. ' data of size ' .. dataset.size)
end

local function make_network()
   local RNN = nn[opt.reclayer:upper()]
   model:add(nn.LookupTable(vocab.size, 200))
   model:add(RNN(200, 200, 100))
   if opt.dropout then model:add(nn.Dropout()) end
   model:add(nn.Linear(100, vocab.size))
   model:add(nn.SoftMax())
   model:add(nn.ClassNLL())
   print(model)
end

local function make_trainer()
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
end

local function accuracy(o, t)
   local max, argmax = o:max(1)
   argmax = argmax:squeeze()
   if argmax == t then
      if opt.debug and training then
         print('CORRECT!', vocab.idx2word[prev], vocab.idx2word[t])
      end
      return 1
   else return 0 end
end

function train()
   training = true
   model:training(true)
   for i = 1, trainData.size do
      try {
         function()
            local ex = trainData.data[i]
            prev = ex.x
            local stats = model:train(ex.x, ex.y, accuracy)
            if opt.debug then
               print(stats.n, stats.cost, stats.accuracy, 
                  stats.forwardTime, stats.backwardTime)
            else
               xlua.progress(i, trainData.size)
            end
         end,
         catch = function(err)
            print(err)
            test()
         end
      }
   end
end

function test()
   training = false
   model:training(false)

   -- Toy example
   local word = 'the'
   local elem = vocab.word2idx[word]
   local output = model:forward(elem).w
   for i = 1, 20 do
      local max, argmax = output:max(1)
      argmax = argmax:squeeze()
      print('the ' .. vocab.idx2word[argmax])
      output[argmax] = 0
   end

   local sum = 0
   for i = 1, testData.size do
      local ex = testData.data[i]
      local output = model:forward(ex.x).w
      if opt.debug then
         sum = sum + accuracy(output, ex.y)
      else
         xlua.progress(i, testData.size)
      end

      if opt.debug and i % 1000 == 0 then
         print(sum / i)
      end
   end

   -- TODO: compute perplexity
end

make_vocab()
make_data(true)
make_data(false)
make_network()
make_trainer()
for epoch = 1, opt.epochs do
   print('Epoch #' .. tostring(epoch))
   train()
   test()
end