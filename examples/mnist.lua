require '../symnn'
print(symnn)

local model = symnn.Sequential()
model:add(symnn.Reshape(784))
model:add(symnn.Linear(784, 100))
model:add(symnn.Sigmoid())
model:add(symnn.Linear(100, 10))

local criterion = symnn.CrossEntropy()
print(model, criterion)

local x = symtorch.Tensor(28, 28):rand(0, 1)
local y = model:forward(x)
print(y.w)

local t = 4
local o, cost = criterion:forward(y, t)
print(o.w, cost)

model:train()