local ClassNLL = Class {
   params = {},

   forward = function(self, input, output)
      self.input = input
      self.output = output
      return output
   end,

   backward = function(self, target)
      local argmax
      if type(target) == 'number' then
         assert(target > 0 and target <= self.input.w:size(1))
         argmax = target
      elseif target.name == 'Tensor' then
         target, argmax = target.w:max(1)
         argmax = argmax:squeeze()
      end

      self.input.dw:copy(self.output.w)
      self.input.dw[argmax] = self.input.dw[argmax] - 1
      return -torch.log(self.output.w[argmax])
   end,

   __tostring = function(self) return 'symnn.ClassNLL' end
}

local MSE = Class {
   params = {},

   forward = function(self, input, output)
      self.input = input
      self.output = output
      return output
   end,

   backward = function(self, target)
      if type(target) == 'number' then
         local argmax = target
         target = torch.zeros(self.input.w:size())
         target[argmax] = 1.0
      end

      local diff = torch.add(self.input.w, -target)
      self.input.dw:copy(diff)
      local cost = diff:pow(2):mean()
      return cost
   end,

   __tostring = function(self) return 'symnn.MSE' end
}

local CrossEntropy = Class {
   params = {},

   __init__ = function(self)
      self.lsm = symtorch.LogSoftMax()
      self.nll = symtorch.ClassNLL()
   end,

   forward = function(self, input)
      self.lsmOutput = self.lsm:forward(input)
      self.nllOutput = self.nll:forward(self.lsmOutput:exp())
      return self.nllOutput
   end,

   backward = function(self, target)
      return self.nll:backward(target)
   end,

   __tostring = function(self) return 'symnn.CrossEntropy' end
}

local Hinge = Class {
   params = {},

   forward = function(self, input, output)
      self.input = input
      self.output = output
      return output
   end,
   
   backward = function(self, target)
      local argmax
      if type(target) == 'number' then
         assert(target > 0 and target <= self.input.w:size(1))
         argmax = target
      elseif target.name == 'Tensor' then
         target, argmax = target.w:max(1)
         argmax = argmax:squeeze()
      end

      local score = self.input.w[argmax]
      local margin = 1.0
      local diff = torch.add(self.input.w, -score):add(margin)
      local cost = diff:sum()
      local grad = diff:gt(0):double()
      self.input.dw:add(grad)
      self.input.dw[argmax] = self.input.dw[argmax] - grad:nElement()
      return cost
   end,

   __tostring = function(self) return 'symnn.Hinge' end
}

return {
   ClassNLL = ClassNLL,
   MSE = MSE,
   CrossEntropy = CrossEntropy,
   Hinge = Hinge,
}