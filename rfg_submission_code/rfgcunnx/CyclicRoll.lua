require 'nn'

local CyclicRoll, parent = torch.class('nn.CyclicRoll','nn.Module')

function CyclicRoll:__init()
   parent.__init(self)
end

function CyclicRoll:updateOutput(input)
	assert(input:dim()==4 and input:size(1)%4 == 0, "input must be 4D tensor and batch size must be divide by 4")
   local sz = input:size()
   sz[2] = sz[2]*4
   self.output:resize(sz)
   input.nn.CyclicRoll_updateOutput(self, input)
   return self.output
end

function CyclicRoll:updateGradInput(input, gradOutput)
   if self.gradInput then
	   self.gradInput:resizeAs(input)
	   input.nn.CyclicRoll_updateGradInput(self, input, gradOutput)
	   return self.gradInput
   end
end
