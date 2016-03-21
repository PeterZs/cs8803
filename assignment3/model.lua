----------------------------------------------------------------------
-- Create CNN and loss to optimize.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
--require 'Dropout' -- Hinton dropout technique


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
local noutputs = 20

-- input dimensions: faces!
local nfeats = 3
local width = 224
local height = 224

-- hidden units, filter sizes (for ConvNet only):
local nstates = {16,32,100}
local filtsize = {5, 7}
local poolsize = {4,4}

----------------------------------------------------------------------
local classifier = nn.Sequential()
local model = nn.Sequential()
local model_name ="model.net"

if opt.model == 'CNN' then
   print(sys.COLORS.red ..  '==> construct CNN')
   ---- TODO --------
   --Create a CNN network as mentioned in the write-up
   --followed by a 2 layer fully connected layers
   --Use ReLU as yoru activations
   local CNN = nn.Sequential()
   CNN:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize[1], filtsize[1]))
   CNN:add(nn.ReLU())
   CNN:add(nn.SpatialMaxPooling(poolsize[1],poolsize[1],poolsize[1],poolsize[1]))
   CNN:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize[2], filtsize[2]))
   CNN:add(nn.ReLU())
   CNN:add(nn.SpatialMaxPooling(poolsize[2],poolsize[2],poolsize[2],poolsize[2]))
   local w = ((width-filtsize[1]+1)/poolsize[1]-filtsize[2]+1)/poolsize[2]
   local h = ((height-filtsize[1]+1)/poolsize[1]-filtsize[2]+1)/poolsize[2]
   local conv_size = nstates[2] * math.floor(w) * math.floor(h)
   CNN:add(nn.Reshape(conv_size))
   CNN:add(nn.Linear(conv_size, nstates[3]))
   CNN:add(nn.ReLU())
   CNN:add(nn.Linear(nstates[3], noutputs))
   classifier:add(nn.LogSoftMax())
   model:add(CNN)
   model:add(classifier)
   model_name = "CNN.net"
elseif opt.model == 'CNN_TANH' then
   print(sys.COLORS.red ..  '==> construct CNN_TANH')
   ---- TODO --------
   --same as above, replace ReLU with Tanh
   local CNN = nn.Sequential()
   CNN:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize[1], filtsize[1]))
   CNN:add(nn.Tanh())
   CNN:add(nn.SpatialMaxPooling(poolsize[1],poolsize[1],poolsize[1],poolsize[1]))
   CNN:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize[2], filtsize[2]))
   CNN:add(nn.Tanh())
   CNN:add(nn.SpatialMaxPooling(poolsize[2],poolsize[2],poolsize[2],poolsize[2]))
   local w = ((width-filtsize[1]+1)/poolsize[1]-filtsize[2]+1)/poolsize[2]
   local h = ((height-filtsize[1]+1)/poolsize[1]-filtsize[2]+1)/poolsize[2]
   local conv_size = nstates[2] * math.floor(w) * math.floor(h)
   CNN:add(nn.Reshape(conv_size))
   CNN:add(nn.Linear(conv_size, nstates[3]))
   CNN:add(nn.Tanh())
   CNN:add(nn.Linear(nstates[3], noutputs))
   classifier:add(nn.LogSoftMax())
   model:add(CNN)
   model:add(classifier)
   model_name = "CNN_TANH.net"
elseif opt.model == 'CNN_DROPOUT' then
   print(sys.COLORS.red ..  '==> construct CNN_DROPOUT')
   ---- TODO --------
   -- Same as CNN but with a Dropout layer
   local CNN = nn.Sequential()
   CNN:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize[1], filtsize[1]))
   CNN:add(nn.ReLU())
   CNN:add(nn.SpatialMaxPooling(poolsize[1],poolsize[1],poolsize[1],poolsize[1]))
   CNN:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize[2], filtsize[2]))
   CNN:add(nn.ReLU())
   CNN:add(nn.SpatialMaxPooling(poolsize[2],poolsize[2],poolsize[2],poolsize[2]))
   local w = ((width-filtsize[1]+1)/poolsize[1]-filtsize[2]+1)/poolsize[2]
   local h = ((height-filtsize[1]+1)/poolsize[1]-filtsize[2]+1)/poolsize[2]
   local conv_size = nstates[2] * math.floor(w) * math.floor(h)
   CNN:add(nn.Reshape(conv_size))
   CNN:add(nn.Dropout())
   CNN:add(nn.Linear(conv_size, nstates[3]))
   CNN:add(nn.ReLU())
   CNN:add(nn.Linear(nstates[3], noutputs))
   classifier:add(nn.LogSoftMax())
   model:add(CNN)
   model:add(classifier)
   model_name = "CNN_DROPOUT.net"
elseif opt.model == 'CNN_FINETUNE' then
   require 'loadcaffe'
   model = loadcaffe.load('models/VGG_CNN_M_deploy.prototxt', 'models/VGG_CNN_M.caffemodel', 'nn')
   ---- TODO --------
   -- model has been loaded with the VGG_CNN_M model
   -- Remove the the last two layer and replace with linear layer to matcb
   -- your requirement followed by a  softmax layer
   --
   
   model_name = "CNN_FINETUNE.net"
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
   model_name = model_name,
}

