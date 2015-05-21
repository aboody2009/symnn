local sigmoid = symtorch.sigmoid
local tanh = symtorch.tanh

local RNN = Class {
   __init__ = function(self, input, hidden, output)
      self.inputSize = input
      self.hiddenSize = hidden
      self.outputSize = output

      self.W_hx = symtorch.Tensor(hidden, input):rand(0, 0.08)
      self.W_hh = symtorch.Tensor(hidden, hidden):rand(0, 0.08)
      self.b_i = symtorch.Tensor(hidden, 1)
      self.W_hy = symtorch.Tensor(hidden, hidden):rand(0, 0.08)
      self.b_y = symtorch.Tensor(hidden, 1)
      self.W_od = symtorch.Tensor(output, hidden):rand(0, 0.08) -- output decoder
      self.b_od = symtorch.Tensor(output)

      self.params = {
         self.W_hx, self.W_hh, self.b_i,
         self.W_hy, self.b_y,
         self.W_od, self.b_od
      }

      self.prev_h = {}
      for i = 1, input do
         table.insert(self.prev_h, symtorch.Tensor(hidden, 1))
      end
   end,

   forward = function(self, input)
      local function step(i, x_t, prev_h)
         -- Modeled after http://arxiv.org/pdf/1409.3215.pdf
         -- First 2 equations of §2
         h_t = sigmoid(self.W_hx:dot(x_t) + self.W_hh:dot(prev_h) + self.b_i)
         y_t = self.W_hy:dot(h_t) + self.b_y
         self.prev_h[i] = h_t
         return h_t, y_t
      end

      local res = symtorch.scan{
         fn = step,
         sequences = {input, self.prev_h},
         nsteps = 1
      }

      local final = res[#res][2]
      return self.W_od:dot(final) + self.b_od
   end,

   __tostring = function(self) return 'symnn.RNN' end
}

local LSTM = Class {
   __init__ = function(self, input, hidden, output)
      self.inputSize  = input
      self.hiddenSize = hidden
      self.outputSize = output

      self.W_xi = symtorch.Tensor(hidden, input):rand(0, 0.08)
      self.W_hi = symtorch.Tensor(hidden, hidden):rand(0, 0.08)
      self.b_i  = symtorch.Tensor(hidden, 1)
      self.W_xf = symtorch.Tensor(hidden, input):rand(0, 0.08)
      self.W_hf = symtorch.Tensor(hidden, hidden):rand(0, 0.08)
      self.b_f  = symtorch.Tensor(hidden, 1)
      self.W_xc = symtorch.Tensor(hidden, input):rand(0, 0.08)
      self.W_hc = symtorch.Tensor(hidden, hidden):rand(0, 0.08)
      self.b_c  = symtorch.Tensor(hidden, 1)
      self.W_xo = symtorch.Tensor(hidden, input):rand(0, 0.08)
      self.W_ho = symtorch.Tensor(hidden, hidden):rand(0, 0.08)
      self.b_o  = symtorch.Tensor(hidden, 1)
      self.W_od = symtorch.Tensor(output, hidden):rand(0, 0.08) -- output decoder
      self.b_od = symtorch.Tensor(output, 1)

      self.params = {
         self.W_xi, self.W_hi, self.b_i,
         self.W_xf, self.W_hf, self.b_f,
         self.W_xc, self.W_hc, self.b_c,
         self.W_xo, self.W_ho, self.b_o,
         self.W_od, self.b_od
      }

      self.prev_h = {}
      self.prev_c = {}
      for i = 1, input do
         table.insert(self.prev_h, symtorch.Tensor(hidden, 1))
         table.insert(self.prev_c, symtorch.Tensor(hidden, 1))
      end
   end,

   forward = function(self, input)
      -- Modeled after http://arxiv.org/pdf/1411.4555v1.pdf
      -- Equations 4-8

      local function step(i, x_t, prev_h, prev_c)
         local i_t = sigmoid(self.W_xi:dot(x_t) + self.W_hi:dot(prev_h) + self.b_i)                   -- (4)
         local f_t = sigmoid(self.W_xf:dot(x_t) + self.W_hf:dot(prev_h) + self.b_f)                   -- (5)
         local c_t = f_t * prev_c + i_t * tanh(self.W_xc:dot(x_t) + self.W_hc:dot(prev_h) + self.b_c) -- (6)
         local o_t = sigmoid(self.W_xo:dot(x_t) + self.W_ho:dot(prev_h) + self.b_o)                   -- (7)
         local h_t = o_t * tanh(c_t)                                                                  -- (8)
         self.prev_h[i] = h_t
         self.prev_c[i] = c_t
         return h_t, c_t
      end

      local res = symtorch.scan{
         fn = step,
         sequences = {input, self.prev_h, self.prev_c},
         nsteps = 1
      }
      local final = res[#res][1]
      return self.W_od:dot(final) + self.b_od
   end,

   __tostring = function(self) return 'symnn.LSTM' end
}

return {
   RNN = RNN,
   LSTM = LSTM
}