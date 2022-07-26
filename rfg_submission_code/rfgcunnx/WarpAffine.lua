require 'nn'

local WarpAffine, parent = torch.class('nn.WarpAffine','nn.Module')

function WarpAffine:__init(outWidth, outHeight)
   parent.__init(self)
   self.outWidth = outWidth
   self.outHeight = outHeight
   self.output = torch.Tensor()
end

function WarpAffine:updateOutput(input)
   assert (#input == 2, "input must be a tensor of input and tensor of warp matrix")
   input[1].nn.WarpAffine_updateOutput(self, input[1], input[2]:float())
   return self.output
end
