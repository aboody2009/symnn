-- elements of this example were taken from: https://github.com/Element-Research/rnn
local nn = require '../symnn'

local batchSize = 8
local hiddenSize = 20
local nIndex = 10000
local sequence = torch.randperm(nIndex)

local offsets = {}
for i=1,batchSize do
   table.insert(offsets, math.ceil(math.random()*batchSize))
end
offsets = torch.DoubleTensor(offsets)

local model = nn.Sequential()
model:add(nn.RNN(batchSize, hiddenSize, 10))
model:add(nn.SoftMax())
model:add(nn.ClassNLL())

local x = symtorch.Tensor(batchSize):copy(offsets)
print(model:forward(x).w)