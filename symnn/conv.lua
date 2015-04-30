local SpatialConvolution = Class {
   __init__ = function(self, nfilters, sx, sy, stride, pad)
      self.filterSize = {nfilters, sx, sy}
      self.W = symtorch.Tensor(nfilters, sx, sy)
      self.b = symtorch.Tensor(nfilters)
      self.params = { self.W, self.b }

      self.stride = stride or 1
      self.pad = pad or 0
   end,

   forward = function(self, input)
      local conv = symtorch.conv2d(input, self.W, self.stride, self.pad)
      local b = symtorch.Tensor(conv:size())
      for i = 1, self.filterSize[1] do
         b[i]:fill(self.b[i])
      end
      return conv + b
   end
}

local SpatialMaxPooling = Class {
   params = {},
   
   __init__ = function(self, sx, sy, stride, pad)
      self.sx = sx or 2
      self.sy = sy or 2
      self.stride = stride or 2
      self.pad = pad or 0
   end,

   forward = function(self, input)
      return symtorch.maxpool2d(input,
         self.sx, self.sy, self.stride, self.pad)
   end
}

return {
   SpatialConvolution = SpatialConvolution,
   SpatialMaxPooling = SpatialMaxPooling
}