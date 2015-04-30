local NonLinear = function(f)
   return Class {
      params = {},
      fn = f,

      forward = function(self, input)
         return self.fn(input)
      end
   }
end

return {
   Tanh = NonLinear(symtorch.tanh),
   Sigmoid = NonLinear(symtorch.sigmoid),
   ReLU = NonLinear(symtorch.relu),
   Softmax = NonLinear(symtorch.softmax)
}