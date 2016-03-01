----------------------------------------------------------------------
-- Create CNN and loss to optimize.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
--require 'Dropout' -- Hinton dropout technique

-- require 'put your custom layer'
require 'MyLinear'
require 'MyLinearSigmoid'
require 'MyLinearSigmoidLinear'
require 'MyTanh'
require 'MySigmoid'
require 'MyReLU'
require 'MyReQU'


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

if opt.model == 'CNN' then
   print(sys.COLORS.red ..  '==> construct CNN')

   local CNN = nn.Sequential()

   -- stage 1: conv+max
   CNN:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize[1], filtsize[1]))
   CNN:add(nn.Threshold())
   CNN:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

   -- stage 2: conv+max
   CNN:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize[2], filtsize[2]))
   CNN:add(nn.Threshold())

   -- stage 3: linear
   classifier:add(nn.Reshape(nstates[2]))
   classifier:add(nn.Linear(nstates[2], 2))

   -- stage 4 : log probabilities
   classifier:add(nn.LogSoftMax())

   for _,layer in ipairs(CNN.modules) do
       if layer.bias then
          layer.bias:fill(.2)
          if i == #CNN.modules-1 then
             layer.bias:zero()
          end
       end
   end

   model:add(CNN)
   model:add(classifier)

elseif opt.model == 'MyLinear' then

   -- stage 1: linear
   classifier:add(nn.Reshape(nfeats*width*height))
   classifier:add(nn.MyLinear(nfeats*width*height, 2))

   -- stage 2 : log probabilities
   classifier:add(nn.LogSoftMax())

   model:add(classifier)

elseif opt.model == 'MyLinearSigmoid' then

   -- stage 1: linear + sigmoid
   classifier:add(nn.Reshape(nfeats*width*height))
   classifier:add(nn.MyLinearSigmoid(nfeats*width*height, 2))

   model:add(classifier)

elseif opt.model == 'MyLinearSigmoidLinear' then

   -- stage 1: two linear layers
   classifier:add(nn.Reshape(nfeats*width*height))
   classifier:add(nn.MyLinearSigmoidLinear(nfeats*width*height, width, 2))

   -- stage 2 : log probabilities
   classifier:add(nn.LogSoftMax())

   model:add(classifier)

elseif opt.model == 'mlp_tanh' then

   classifier:add(nn.Reshape(nfeats*width*height))

   -- stage 1: two linear layers
   classifier:add(nn.Linear(nfeats*width*height, width))
   -- TODO --------------------------------------------
   -- add your tanh layer
   classifier:add(nn.MyTanh())
   ----------------------------------------------------
   classifier:add(nn.Linear(width,2))

   -- stage 2 : log probabilities
   classifier:add(nn.LogSoftMax())

   model:add(classifier)

elseif opt.model == 'mlp_sigmoid' then

   classifier:add(nn.Reshape(nfeats*width*height))

   -- stage 1: two linear layers
   classifier:add(nn.Linear(nfeats*width*height, width))
   -- TODO --------------------------------------------
   -- add your Sigmoid layer
   classifier:add(nn.MySigmoid())
   ----------------------------------------------------
   classifier:add(nn.Linear(width,2))

   -- stage 2 : log probabilities
   classifier:add(nn.LogSoftMax())

   model:add(classifier)

elseif opt.model == 'mlp_relu' then

   classifier:add(nn.Reshape(nfeats*width*height))

   -- stage 1: two linear layers
   classifier:add(nn.Linear(nfeats*width*height, width))
   -- TODO --------------------------------------------
   -- add your ReLU layer
   classifier:add(nn.MyReLU())
   ----------------------------------------------------
   classifier:add(nn.Linear(width,2))

   -- stage 2 : log probabilities
   classifier:add(nn.LogSoftMax())

   model:add(classifier)

elseif opt.model == 'mlp_requ' then

   classifier:add(nn.Reshape(nfeats*width*height))

   -- stage 1: two linear layers
   classifier:add(nn.Linear(nfeats*width*height, width))
   -- TODO --------------------------------------------
   -- add your ReQU layer
   classifier:add(nn.MyReQU())
   ----------------------------------------------------
   classifier:add(nn.Linear(width,2))

   -- stage 2 : log probabilities
   classifier:add(nn.LogSoftMax())

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

