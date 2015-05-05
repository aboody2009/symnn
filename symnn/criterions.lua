local ClassNLL = Class {
   forward = function(self, input, output, target)
      -- input was input to softmax
      -- output was output of softmax
      -- target is target
      local argmax
      if type(target) == 'number' then
         assert(target > 0 and target <= input.w:size(1))
         argmax = target
      elseif target.name == 'Tensor' then
         target, argmax = target.w:max(1)
         argmax = argmax:squeeze()
      end

      input.dw:copy(output.w)
      input.dw[argmax] = input.dw[argmax] - 1
      return -torch.log(output.w[argmax])
   end,

   __tostring = function(self) return 'symnn.ClassNLL' end
}

local MSE = Class {
   forward = function(self, input, target)
      if type(target) == 'number' then
         target = symtorch.Tensor(input.w:size()):fill(target)
      end

      local diff = torch.add(input.w, -target.w)
      input.dw:copy(diff)
      local cost = diff:pow(2):mean()
      return cost
   end,

   __tostring = function(self) return 'symnn.MSE' end
}

local CrossEntropy = Class {
   forward = function(self, input, target)
      local output = symtorch.softmax(input)
      print(output.w)
      output.w:log() -- LogSoftMax

      local argmax
      if type(target) == 'number' then
         assert(target > 0 and target <= input.w:size(1))
         argmax = target
      elseif target.name == 'Tensor' then
         target, argmax = target.w:max(1)
         argmax = argmax:squeeze()
      end

      local cost = -output.w[argmax]
      input.dw:copy(output.w)
      input.dw[argmax] = input.dw[argmax] - 1
      return output, cost
   end,

   __tostring = function(self) return 'symnn.CrossEntropy' end
}

local Hinge = Class {
   forward = function(self, input, target)
      local argmax
      if type(target) == 'number' then
         assert(target > 0 and target <= input.w:size(1))
         argmax = target
      elseif target.name == 'Tensor' then
         target, argmax = target.w:max(1)
         argmax = argmax:squeeze()
      end

      local score = input.w[argmax]
      local margin = 1.0
      local diff = torch.add(input.w, -score):add(margin)
      local cost = diff:sum()
      local grad = diff:gt(0):double()
      input.dw:add(grad)
      input.dw[argmax] = input.dw[argmax] - grad:nElement()
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