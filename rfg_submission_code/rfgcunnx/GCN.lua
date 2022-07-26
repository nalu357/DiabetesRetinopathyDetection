require 'nn'

-- global contrast normalization
local GCN, parent = torch.class('nn.GCN','nn.Module')

function GCN:__init(byChannel)
   parent.__init(self)
   if byChannel == nil then
	   self.byChannel = true
   else
	   self.byChannel = byChannel
   end
end

function GCN:updateOutput(input)
   local nDim = input:dim()
   assert(nDim == 3 or nDim == 4, "input must be 3D or 4D tensor")
   local out 
   local sz = input:size()
   if (nDim == 3) then
   	  if self.byChannel then
		  out = input:view(sz[1], sz[2]*sz[3])
	  else
		  out = input:view(1, sz[1]*sz[2]*sz[3])
	  end
   else
   	  if self.byChannel then
		  out = input:view(sz[1]*sz[2], sz[3]*sz[4])
	  else
		  out = input:view(sz[1], sz[2]*sz[3]*sz[4])
	  end
   end
   local mean  = out:mean(2)
   local std  = out:std(2)
   std:add(1e-12)
   out:add(-1, mean:expandAs(out))
   out:cdiv(std:expandAs(out))
   self.output = out:viewAs(input)
   return self.output
end
