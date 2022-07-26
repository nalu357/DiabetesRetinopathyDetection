require 'nn'

--create a Tensor by rotating the original tensor 0, 90, 180, 270 counter clock wise and stack them together as output

local CyclicSlice, parent = torch.class('nn.CyclicSlice','nn.Module')

function CyclicSlice:__init()
   parent.__init(self)
end

function CyclicSlice:updateOutput(input)
   local sz = input:size()
   local nDim = input:dim()
   assert (nDim == 3 or nDim ==4, "input must have 3 or 4 dimensions")
   if (nDim == 3) then
	   self.output:resize(4, unpack(sz:totable()))
   else
	   sz[1] = sz[1]*4
	   self.output:resize(sz)
   end
   input.nn.CyclicSlice_updateOutput(self, input)
   return self.output
end

function CyclicSlice:updateGradInput(input, gradOutput)
   if self.gradInput then
       assert(gradOutput:nElement() == 4*input:nElement(), "gradOutput must have 4 times input elements")
	   input.nn.CyclicSlice_updateGradInput(self, input, gradOutput)
	   return self.gradInput
   end
end
