local Linear = Class {
   __init__ = function(self, inputSize, outputSize)
      self.W = symtorch.Tensor(outputSize, inputSize):rand()
      self.b = symtorch.Tensor(outputSize)
      self.params = {self.W, self.b}
   end,

   forward = function(self, input)
      return self.W:dot(input) + self.b
   end,

   __tostring = function(self) return 'symnn.Linear' end
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
   end,

   __tostring = function(self) return 'symnn.Reshape' end
}

local Identity = Class {
   params = {},
   forward = function(self, input) return input end,
   __tostring = function(self) return 'symnn.Identity' end
}

local Dropout = Class {
   params = {},
   training = true,

   __init__ = function(self, p)
      self.p = p or 0.5
   end,

   forward = function(self, input)
      if self.training then
         local dist = symtorch.rng.binomial(input.w:size(), 1 - p)
         return input * dist
      else
         return input
      end
   end,

   __tostring = function(self) return 'symnn.Dropout' end
}

return {
   Linear = Linear,
   Reshape = Reshape,
   Identity = Identity,
   Dropout = Dropout
}