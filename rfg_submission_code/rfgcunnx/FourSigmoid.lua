local FourSigmoid, parent = torch.class('nn.FourSigmoid', 'nn.Module')

function FourSigmoid:__init(alpha)
	parent.__init(self)
	self.alpha = alpha or 10
end

function FourSigmoid:updateOutput(input)
   input.nn.FourSigmoid_updateOutput(self, input)
   self.output:add(1)
   return self.output
end

function FourSigmoid:updateGradInput(input, gradOutput)
   input.nn.FourSigmoid_updateGradInput(self, input, gradOutput)
   return self.gradInput
end
