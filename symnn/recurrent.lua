local sigmoid = symtorch.sigmoid
local tanh = symtorch.tanh

local RNN = Class {
   __init__ = function(self, input, hidden, output)
      self.inputSize = input
      self.hiddenSize = hidden
      self.outputSize = output

      self.W_ih = symtorch.Tensor(hidden, input):rand(0, 0.08)
      self.b_ih = symtorch.Tensor(hidden, 1)
      self.W_hh = symtorch.Tensor(hidden, hidden):rand(0, 0.8)
      self.b_hh = symtorch.Tensor(hidden, 1)
      self.W_od = symtorch.Tensor(output, hidden):rand(0, 0.8) -- output decoder
      self.b_od = symtorch.Tensor(output)

      self.params = {
         self.W_ih, self.b_ih,
         self.W_hh, self.b_hh,
         self.W_od, self.b_od
      }
   end,

   forward = function(self, input)
      local function step(prev_x, prev_h)
         h_t = tanh(self.W_ih:dot(prev_x) + self.W_hh:dot(prev_h) + self.b_ih)
         y_t = self.W_hh:dot(h_t) + self.b_hh
         return h_t, y_t
      end

      local mem = symtorch.Tensor(self.hiddenSize, 1)
      local res = symtorch.scan{fn=step, sequences={input, mem}}
      local final = res[#res][1]
      return self.W_od:dot(final) + self.b_od
   end
}

local LSTM = Class {
   __init__ = function(self, input, hidden, output)
      self.inputSize  = input
      self.hiddenSize = hidden
      self.outputSize = output

      self.W_hi = symtorch.Tensor(hidden, input):rand(0, 0.08)
      self.W_ci = symtorch.Tensor(hidden, hidden):rand(0, 0.08)
      self.b_i  = symtorch.Tensor(hidden, 1)
      self.W_hf = symtorch.Tensor(hidden, input):rand(0, 0.08)
      self.W_cf = symtorch.Tensor(hidden, hidden):rand(0, 0.08)
      self.b_f  = symtorch.Tensor(hidden, 1)
      self.W_hc = symtorch.Tensor(hidden, input):rand(0, 0.08)
      self.b_c  = symtorch.Tensor(hidden, 1)
      self.W_ho = symtorch.Tensor(hidden, input):rand(0, 0.08)
      self.W_co = symtorch.Tensor(hidden, hidden):rand(0, 0.08)
      self.b_o  = symtorch.Tensor(hidden, 1)
      self.W_od = symtorch.Tensor(output, hidden):rand(0, 0.08) -- output decoder
      self.b_od = symtorch.Tensor(output, 1)

      self.params = {
         self.W_hi, self.W_ci, self.b_i,
         self.W_hf, self.W_cf, self.b_f,
         self.W_hc, self.b_c,
         self.W_ho, self.W_co, self.b_o,
         self.W_od, self.b_od
      }
   end,

   forward = function(self, input)
      -- Modeled after
      -- https://www.cs.toronto.edu/~hinton/absps/RNN13.pdf
      -- Equations 3-7

      local function step(prev_h, prev_c)
         local i_t = sigmoid(self.W_hi:dot(prev_h) + self.W_ci:dot(prev_c) + self.b_i) -- (3)
         local f_t = sigmoid(self.W_hf:dot(prev_h) + self.W_cf:dot(prev_c) + self.b_f) -- (4)
         local c_t = f_t * prev_c + i_t * tanh(self.W_hc:dot(prev_h) + self.b_c)       -- (5)
         local o_t = sigmoid(self.W_ho:dot(prev_h) + self.W_co:dot(prev_c) + self.b_o) -- (6)
         local h_t = o_t * tanh(c_t)                                                   -- (7)
         return h_t, c_t
      end

      local mem = symtorch.Tensor(self.hiddenSize, 1)
      local res = symtorch.scan{fn=step, sequences={input, mem}}
      local final = res[#res][1]
      return self.W_od:dot(final) + self.b_od
   end
}

return {
   RNN = RNN,
   LSTM = LSTM
}