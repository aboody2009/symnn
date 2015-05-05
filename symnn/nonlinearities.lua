local NonLinear = function(f, name)
   return Class {
      params = {},
      fn = f,

      forward = function(self, input)
         return self.fn(input)
      end,

      __tostring = function(self)
         return 'symnn.' .. name
      end
   }
end

local LogSoftMax = Class {
   params = {},

   forward = function(self, input)
      return symtorch.log(symtorch.softmax(input))
   end,

   __tostring = function(self) return 'symnn.LogSoftMax' end
}

return {
   Tanh = NonLinear(symtorch.tanh, 'Tanh'),
   Sigmoid = NonLinear(symtorch.sigmoid, 'Sigmoid'),
   ReLU = NonLinear(symtorch.relu, 'ReLU'),
   Exp = NonLinear(symtorch.exp, 'Exp'),
   Log = NonLinear(symtorch.log, 'Log'),
   SoftMax = NonLinear(symtorch.softmax, 'Softmax'),
   LogSoftMax = LogSoftMax
}