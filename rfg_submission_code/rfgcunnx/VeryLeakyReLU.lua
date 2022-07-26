local VeryLeakyReLU, parent = torch.class('nn.VeryLeakyReLU','nn.Module')

function VeryLeakyReLU:__init(th,alpha)
   parent.__init(self)
   self.threshold = th or 1e-6
   self.alpha = alpha or 0
   if (th and type(th) ~= 'number') or (alpha and type(alpha) ~= 'number') then
      error('nn.VeryLeakyReLU(threshold, alpha)')
   end
end

function VeryLeakyReLU:updateOutput(input)
   input.nn.VeryLeakyReLU_updateOutput(self, input)
   return self.output
end

function VeryLeakyReLU:updateGradInput(input, gradOutput)
   input.nn.VeryLeakyReLU_updateGradInput(self, input, gradOutput)
   return self.gradInput
end
