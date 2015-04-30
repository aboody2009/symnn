local Linear = Class {
   __init__ = function(self, inputSize, outputSize)
      self.W = symtorch.Tensor(outputSize, inputSize):rand()
      self.b = symtorch.Tensor(outputSize)
      self.params = {self.W, self.b}
   end,

   forward = function(self, input)
      return self.W:dot(input) + self.b
   end
}

local Reshape = Class {
   params = {},

   __init__ = function(self, shape)
      self.shape = shape
   end,

   forward = function(self, input)
      input.w:resize(self.shape)
      input.dw:resize(self.shape)
      return input
   end
}

local Identity = Class {
   params = {},
   forward = function(self, input) return input end
}

local Dropout = Class {
   params = {},
   training = true,

   __init__ = function(self, p)
      self.p = p or 0.5
   end,

   forward = function(self, input)
      if self.training then
         local dist = symtorch.rng.binomial(input:size(), 1 - p)
         return input * dist
      else
         return input
      end
   end
}

return {
   Linear = Linear,
   Reshape = Reshape,
   Identity = Identity,
   Dropout = Dropout
}