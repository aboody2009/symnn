local LookupTable = Class {
   __init__ = function(self, nIndex, nOuput)
      self.W = symtorch.Tensor(nIndex, nOuput):rand()
      self.params = { self.W }
   end,

   forward = function(self, input)
      return self.W:index(input)
   end,

   __tostring = function(self) return 'symnn.LookupTable' end
}

return { LookupTable = LookupTable }