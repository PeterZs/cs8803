----------------------------------------------------------------------
-- Create CNN and loss to optimize.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
--require 'Dropout' -- Hinton dropout technique

-- TODO ---------------------------------------------
-- require 'put your custom layer'
require 'MyLinear'
require 'MyLinearSigmoid'
require 'MyLinearSigmoidLinear'


----------------------------------------------------

if opt.type == 'cuda' then
   nn.SpatialConvolutionMM = nn.SpatialConvolution
end

----------------------------------------------------------------------
print '==> processing options'

--[[
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
-- model:
cmd:option('-model', 'CNN', 'type of model to construct: MyLinear | MyLinearSigmoid | MyLinearSigmoidLinear | CNN')
cmd:text()
opt = cmd:parse(arg or {})
--]]

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> define parameters')

-- 2-class problem: faces!
local noutputs = 2

-- input dimensions: faces!
local nfeats = 1
local width = 32
local height = 32

-- hidden units, filter sizes (for ConvNet only):
local nstates = {16,32}
local filtsize = {5, 7}
local poolsize = 4

----------------------------------------------------------------------
local classifier = nn.Sequential()
local model = nn.Sequential()

if opt.model == 'MyLinear' then

   -- stage 1: linear
   -- TODO --------------------------------------------
   -- Create your linear model
   -- classifier:add(xxx)
   classifier:add(nn.Reshape(opt.batchSize,width*height*nfeats,false))
   classifier:add(nn.MyLinear(width*height*nfeats,noutputs))

   ----------------------------------------------------

   -- stage 2 : log probabilities
   classifier:add(nn.LogSoftMax())

   model:add(classifier)

elseif opt.model == 'MyLinearSigmoid' then

   -- stage 1: linear + sigmoid
   -- TODO --------------------------------------------
   -- Create your linear + sigmoid model
   -- classifier:add(xxx)
   classifier:add(nn.Reshape(opt.batchSize,width*height*nfeats,false))
   classifier:add(nn.MyLinearSigmoid(width*height*nfeats,noutputs))

   ----------------------------------------------------

   -- stage 2 : log probabilities
   classifier:add(nn.LogSoftMax())
   model:add(classifier)

elseif opt.model == 'MyLinearSigmoidLinear' then

   -- stage 1: two linear layers
   -- TODO --------------------------------------------
   -- Create your linear + sigmoid + linear model
   -- classifier:add(xxx)
   classifier:add(nn.Reshape(opt.batchSize,width*height*nfeats,false))
   classifier:add(nn.MyLinearSigmoidLinear(width*height*nfeats,width,noutputs))

   ----------------------------------------------------

   -- stage 2 : log probabilities
   -- classifier:add(nn.LogSoftMax())

   model:add(classifier)
end

-- Loss: NLL
loss = nn.ClassNLLCriterion()


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> here is the network:')
print(model)

if opt.type == 'cuda' then
   model:cuda()
   loss:cuda()
end

-- return package:
return {
   model = model,
   loss = loss,
}

