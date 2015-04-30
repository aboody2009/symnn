return {
   Sequential = Class {
      layers = {},
      params = {},
      isTraining = true,

      add = function(self, layer)
         table.insert(self.layers, layer)
         for i = 1, #layer.params do
            table.insert(self.params, layer.params[i])
         end
      end,

      forward = function(self, input)
         local output = input
         for i = 1, #self.layers do
            output = self.layers[i]:forward(output)
         end
         symtorch.update(self.params)
         return output
      end,

      training = function(self, val)
         self.isTraining = val
         symtorch._graph.needs_backprop = val
      end,

      __tostring = function()
         return 'Sequential'
      end
   }
}