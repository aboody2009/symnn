function iterator(f)
   return function(...)
      local d = f(...)
      local n = 0
      local size = #d
      return function()
         n = n + 1
         if (d and n <= size) then return d[n]
         else return nil end
      end
   end
end

string.gsplit = iterator(string.split)

function try(options)
   local block = options.block or options.try or options[1] or function() end
   local catch = options.catch or options[2] or function(err) end
   local finally = options.finally or options[3] or function() end
   local ok, err = pcall(block)
   if err ~= nil then catch(err) end
   finally()
end