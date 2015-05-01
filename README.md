# symnn

Neural Networks for [symtorch](https://github.com/benglard/symtorch).

As much as symtorch is still a work-in-progress, symnn is a work-in-progress even more so.

## TODO:
* Cost layers/criterions (ClassNLL, MSE, CrossEntropy, Hinge)
* Test and get everything actually working
* Examples
  1. MNIST
  2. Autoencoder like
  3. Image captioning like (will scan work for rnn/lstm?)
* Batched training
* LeakyReLU/PReLU
* Embedding layers
* preprocessing
* more training/optim methods
* even more

## (Future) Example Usage

```lua
require 'symnn'
local model = symnn.Sequential()
model:add(symnn.Reshape(784))
model:add(symnn.Linear(784, 100))
model:add(symnn.Sigmoid())
model:add(symnn.Linear(100, 10))
model:add(symnn.Softmax())
print(model)

local x = symtorch.Tensor(28, 28):rand(0, 1)
local y = model:forward(x)
print(y.w)
```