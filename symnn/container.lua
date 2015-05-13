local Window = Class {
   -- Keep windowed stats, averages

   __init__ = function(self, options)
      options = options or {}
      self.size = options.size or 100
      self.minsize = options.minsize or 20
      self.v = {}
      self.sum = 0
   end,

   add = function(self, val)
      table.insert(self.v, val)
      self.sum = self.sum + val
      if #self.v > self.size then
         local victim = table.remove(self.v, 1)
         self.sum = self.sum - victim
      end
   end,

   average = function(self)
      if #self.v < self.minsize then return -1
      else return self.sum / #self.v
      end
   end,

   reset = function(self)
      self.v = {}
      self.sum = 0
   end
}

local Sequential = Class {
   layers = {},
   params = {},
   isTraining = true,
   nsteps = 0,
   win = Window(),

   __init__ = function(self, batchSize)
      self.batchSize = batchSize or 1
   end,

   add = function(self, layer)
      table.insert(self.layers, layer)
      for i = 1, #layer.params do
         table.insert(self.params, layer.params[i])
      end
   end,

   forward = function(self, input)
      local output = nil
      for i = 1, #self.layers - 1 do
         input = output or input
         output = self.layers[i]:forward(input)
      end
      return self.layers[#self.layers]:forward(input, output)
   end,

   backward = function(self, target)
      local cost = self.layers[#self.layers]:backward(target)
      _graph:backward()
      return cost
   end,

   train = function(self, input, target, accuracy)
      self.nsteps = self.nsteps + 1

      local t = torch.tic()
      local output = self:forward(input).w
      local t2 = torch.tic()
      local ftime = t2 - t

      t = torch.tic()
      local cost = self:backward(target)
      t2 = torch.tic()
      local btime = t2 - t

      if accuracy ~= nil then
         self.win:add(accuracy(output, target))
      end

      if self.nsteps % self.batchSize == 0 then
         symtorch.update(self.params)
      end

      return {
         output = output,
         cost = cost,
         forwardTime = ftime,
         backwardTime = btime,
         accuracy = self.win:average()
      }
   end,

   training = function(self, val)
      self.isTraining = val
      _graph.needsBackprop = val
      for i = 1, #self.layers do
         if self.layers[i].training ~= nil then
            self.layers[i].training = val
         end
      end
   end,

   __len = function(self) return #self.layers end,

   __tostring = function(self)
      local tab = '  '
      local line = '\n'
      local next = ' -> '
      local str = 'symnn.Sequential {' .. line .. tab .. '[input'
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

return { Sequential = Sequential }