require 'nn'

local CyclicPool, parent = torch.class('nn.CyclicPool','nn.Module')

function CyclicPool:__init()
   parent.__init(self)
end

function CyclicPool:updateOutput(input)
   local sz = input:size()
   local nDim = input:dim()
   assert (nDim ==4, "input must have 4 dimensions")
   assert (sz[1]%4 == 0, "batch size must be divided by 4")
   sz[1] = sz[1]/4
   self.output:resize(sz)
   input.nn.CyclicPool_updateOutput(self, input)
   return self.output
end

function CyclicPool:updateGradInput(input, gradOutput)
   if self.gradInput then
       assert(gradOutput:nElement()*4 == input:nElement(), "number of element in input must be 4 times of gradoutput")
	   input.nn.CyclicPool_updateGradInput(self, input, gradOutput)
	   return self.gradInput
   end
end
