require "cutorch"
require "nn"
require "cunn"
require "nnx"
require "librfgcunnx"

torch.include('rfgcunnx', 'VeryLeakyReLU.lua')
torch.include('rfgcunnx', 'FracSpatialMaxPooling.lua')
torch.include('rfgcunnx', 'CyclicSlice.lua')
torch.include('rfgcunnx', 'CyclicPool.lua')
torch.include('rfgcunnx', 'CyclicRoll.lua')
torch.include('rfgcunnx', 'WarpAffine.lua')
torch.include('rfgcunnx', 'GCN.lua')
torch.include('rfgcunnx', 'FourSigmoid.lua')
