# symnn

Neural Networks for [symtorch](https://github.com/benglard/symtorch).

As much as symtorch is still a work-in-progress, symnn is a work-in-progress even more so. It all almost works, but I would not recommend using as of now.

## TODO:
* Test and get everything actually working
* Examples
  1. MNIST
  2. Autoencoder like
  3. Image captioning like? (will scan work for rnn/lstm?)
* Batched training
* LeakyReLU/PReLU
* Embedding layers
* preprocessing
* more training/optim methods
* even more

## Example Usage

```lua
require 'symnn'
local model = symnn.Sequential()
model:add(symnn.Reshape(784))
model:add(symnn.Linear(784, 100))
model:add(symnn.ReLU())
model:add(symnn.Linear(100, 10))
model:add(symnn.SoftMax())
model:add(symnn.ClassNLL())
print(model)

local x = symtorch.Tensor(28, 28):rand()
local y = model:forward(x)
local target = 5
local cost = model:backward(y)
```

## Installation

```
> (sudo) luarocks install https://raw.githubusercontent.com/benglard/luaclass/master/luaclass-scm-1.rockspec
> (sudo) luarocks install https://raw.githubusercontent.com/benglard/luaimport/master/luaimport-scm-1.rockspec
> (sudo) luarocks install https://raw.githubusercontent.com/benglard/symtorch/master/symtorch-scm-1.rockspec
> (sudo) luarocks install https://raw.githubusercontent.com/benglard/symnn/master/symnn-scm-1.rockspec
```