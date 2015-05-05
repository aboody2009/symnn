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
         return output
      end,

      train = function(self)
         symtorch.update(self.params)
      end,

      training = function(self, val)
         self.isTraining = val
         _graph.needsBackprop = val
      end,

      __tostring = function(self)
         local tab = '  '
         local line = '\n'
         local next = ' -> '
         local str = 'nn.Sequential {' .. line .. tab .. '[input'
         for i = 1, #self.layers do
            str = str .. next .. '(' .. i .. ')'
         end
         str = str .. next .. 'output]'
         for i = 1, #self.layers do
            str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.layers[i]):gsub(line, line .. tab)
         end
         str = str .. line .. '}'
         return str
      end
   }
}