local sgd = Class {
   stepCache = {},

   __init__ = function(self, options)
      options = options or {}
      self.lr = options.lr or 0.01
      self.momentum = options.momentum or 0.9
   end,

   __call = function(self, params)
      for i = 1, #params do
         local p = params[i]
         p.dw:mul(-self.lr)

         if self.momentum > 0.0 then
            if self.stepCache[i] == nil then
               self.stepCache[i] = symtorch.Tensor(p.w:size())
            end
            local s = self.stepCache[i]

            s.w:mul(self.momentum):add(p.dw)
            p.w:add(s.w)
         else
            p.w:add(p.dw)
         end

         p.dw:zero()
      end
   end,
}

return {
   sgd = sgd,
   rmsprop = symtorch.update
}