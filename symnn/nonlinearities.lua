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

return {
   Tanh = NonLinear(symtorch.tanh, 'Tanh'),
   Sigmoid = NonLinear(symtorch.sigmoid, 'Sigmoid'),
   ReLU = NonLinear(symtorch.relu, 'ReLU'),
   Softmax = NonLinear(symtorch.softmax, 'Softmax')
}