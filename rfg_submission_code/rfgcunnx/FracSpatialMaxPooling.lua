local FracSpatialMaxPooling, parent = torch.class('nn.FracSpatialMaxPooling', 'nn.Module')

--if fixed, it will not change the grid every iteration
function FracSpatialMaxPooling:__init(fraction, mode, olap, fixed)
   parent.__init(self)
--mode 0 is pseudo random
--mode 1 is random
   assert (fraction > 1 and fraction < 2, "fraction must be between 1 and 2")
   self.fraction = fraction
   self.mode = mode or  0
   self.col_ind_start = torch.Tensor()  -- indices for the width dimension
   self.col_ind_end = torch.Tensor()  -- indices for the width dimension
   self.row_ind_start = torch.Tensor()  -- indices for the height dimension
   self.row_ind_end = torch.Tensor()  -- indices for the height dimension
   self.indices = torch.Tensor()
   if (olap ~= nil) then
	   self.olap = olap
   else
	   self.olap = true
   end
   self.fixed = fixed or false
end

function FracSpatialMaxPooling:genSeq_bg(n, alpha, start_ind, end_ind)
	local nout = math.floor(n/alpha)
	start_ind:resize(nout)
	end_ind:resize(nout)

	local alpha1 = (n-2)/(nout-1)

	if self.mode == 0 then -- pseudo random overlapping
		local u = torch.uniform(0, 10000)
		local s = torch.range((u+1)*alpha1, alpha1*(nout+u+1), alpha1)
		s:floor()
		local b0 = torch.zeros(nout)
		for i=1,nout-1 do
			b0[i+1] = b0[i]+s[i+1] - s[i]
		end
		b0[nout] = n-2
		local b1 = torch.add(b0 , 2)
		start_ind:copy(b0)
		end_ind:copy(b1)
	else -- random overlapping
		local tmp1 = {}
		for i = 0,2*nout-n do 
			table.insert(tmp1, 1)
		end
		for i = 0,n-nout-1 do 
			table.insert(tmp1, 2)
		end

		local ind = torch.randperm(nout-1)
		local b0 = torch.zeros(nout)

		b0[1] = 1
	    for i = 1,nout-1 do 
		    b0[i+1] = tmp1[ind[i]] + b0[i]
		end

		local b1 = torch.add(b0, 2)
		start_ind:copy(b0)
		end_ind:copy(b1)
	end

	return nout
end

function FracSpatialMaxPooling:genSeq_jx(n, alpha, out)
	local nout = math.floor(n/alpha)
	out:resize(nout+1)

	if self.mode == 0 then
		local min_u, max_u
		max_u = n/alpha - nout
		min_u = 0
		if nout == math.floor((n-1)/alpha) then
		   min_u = (n-1)/alpha - nout
		else
		   max_u = math.min(max_u, (n-1)/alpha - nout +1)
		end
		local u = torch.uniform(min_u, max_u)
		local tmp = torch.range(u, alpha*(nout+u), alpha)
	    tmp:floor()
		out:copy(tmp)
	else
	    local a = math.floor(alpha)
	    local b = math.ceil(alpha)
	    local nb = n-nout*a
	    local na = nout - nb
		local tmp = {}
		for i = 1, na do
		    table.insert(tmp, a)
	    end
		for i = 1, nb do
		    table.insert(tmp, b)
	    end

		local ind = torch.randperm(nout)
		local tmp1 = torch.Tensor(nout+1)
	    tmp1[1] = 0
	    for i = 1,nout do 
		    tmp1[i+1] = tmp[ind[i]] + tmp1[i]
		end
		out:copy(tmp1)
	end

	return nout
end

function FracSpatialMaxPooling:updateOutput(input)
   local input_dim = input:nDimension()
   assert (input_dim ~=3 or input_dim ~=4, "input dimension must be 3 or 4")
   local output_rows, output_cols
   local in_rows, in_cols
   if (input_dim ==4 ) then
      in_rows = input:size(3)
      in_cols = input:size(4)
   else
      in_rows = input:size(2)
      in_cols = input:size(3)
   end

   if self.fixed then
	   output_rows = self.output_rows or self:genSeq_bg(in_rows, self.fraction, self.row_ind_start, self.row_ind_end)
	   output_cols = self.output_cols or self:genSeq_bg(in_cols, self.fraction, self.col_ind_start, self.col_ind_end)
	   self.output_rows = output_rows
	   self.output_cols = output_cols
   else
	   output_rows = self:genSeq_bg(in_rows, self.fraction, self.row_ind_start, self.row_ind_end)
	   output_cols = self:genSeq_bg(in_cols, self.fraction, self.col_ind_start, self.col_ind_end)
   end
   --print (output_rows .."   "..output_cols) 
   --print (in_rows .."   "..in_cols) 
   if (input_dim ==4 ) then
       self.output:resize(input:size(1), input:size(2), output_rows, output_cols)
   else
       self.output:resize(input:size(1), output_rows, output_cols)
   end

   local count = self.output:nElement()
   self.indices:resize(count)

   input.nn.FracSpatialMaxPooling_updateOutput(self, input, output_rows,  output_cols)
	--[[
	print ('CCCC')
	print (input:nElement())
	print (self.indices:nElement())
	print (self.indices:max())
	print (self.indices:min())
	]]
   return self.output
end

function FracSpatialMaxPooling:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   input.nn.FracSpatialMaxPooling_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function FracSpatialMaxPooling:_sfreeCaches()
   self.gradInput = torch.Tensor()
   self.output = torch.Tensor()
   self.indices = torch.Tensor()
	collectgarbage()
end

function FracSpatialMaxPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
   self.row_ind_start:storage():resize(0)
   self.col_ind_start:storage():resize(0)
   self.row_ind_end:storage():resize(0)
   self.col_ind_end:storage():resize(0)
end
