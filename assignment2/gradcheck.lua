require 'pl'
require 'trepl'
require 'torch'   
require 'nn'

-- TODO ---------------------------------------------
-- require 'put your custom layer'
require 'MyLinear'
require 'MyLinearSigmoid'
require 'MyLinearSigmoidLinear'
require 'MyTanh'
require 'MySigmoid'
require 'MyReLU'
require 'MyReQU'
----------------------------------------------------

-- SETTINGS
print(sys.COLORS.red ..  '==> processing options')

opt = lapp[[
   -p,--type               (default float)       float or cuda
      --model              (default MyLinear)    network model
]]

--------------------------------------------------------------

function create_model(opt)
  ------------------------------------------------------------------------------
  -- MODEL
  ------------------------------------------------------------------------------
  local n_inputs = 4
  local embedding_dim = 2
  local n_classes = 3

  -- OUR MODEL:
  --     linear -> sigmoid -> linear -> softmax
  local model = nn.Sequential()

  if opt.model == 'MyLinear' then
    model:add(nn.MyLinear(n_inputs, n_classes))
  elseif opt.model == 'MyLinearSigmoid' then
    model:add(nn.MyLinearSigmoid(n_inputs, n_classes))
  elseif opt.model == 'MyLinearSigmoidLinear' then
    model:add(nn.MyLinearSigmoidLinear(n_inputs, embedding_dim, n_classes))
  elseif opt.model == 'mlp_tanh' then
    model:add(nn.Linear(n_inputs, embedding_dim))
    -- TODO --------------------------------------------
   model:add(nn.MyTanh())
    ----------------------------------------------------
    model:add(nn.Linear(embedding_dim,n_classes))
  elseif opt.model == 'mlp_sigmoid' then
    model:add(nn.Linear(n_inputs, embedding_dim))
    -- TODO --------------------------------------------
   model:add(nn.MySigmoid())
    ----------------------------------------------------
    model:add(nn.Linear(embedding_dim,n_classes))
  elseif opt.model == 'mlp_relu' then
    model:add(nn.Linear(n_inputs, embedding_dim))
    -- TODO --------------------------------------------
   model:add(nn.MyReLU())
    ----------------------------------------------------
    model:add(nn.Linear(embedding_dim,n_classes))
  elseif opt.model == 'mlp_requ' then
    model:add(nn.Linear(n_inputs, embedding_dim))
    -- TODO --------------------------------------------
   model:add(nn.MyReQU())
    ----------------------------------------------------
    model:add(nn.Linear(embedding_dim,n_classes))
  end

  --model:add(nn.LogSoftMax())

  ------------------------------------------------------------------------------
  -- LOSS FUNCTION
  ------------------------------------------------------------------------------
  local criterion = nn.ClassNLLCriterion()

  return model, criterion
end

--------------------------------------------------------------

-- function that numerically checks gradient of the loss:
-- f is the scalar-valued function
-- g returns the true gradient (assumes input to f is a 1d tensor)
-- returns difference, true gradient, and estimated gradient
local function checkgrad(f, g, x, eps)
  -- compute true gradient
  local grad = g(x)

  -- compute numeric approximations to gradient
  local eps = eps or 1e-7
  local grad_est = torch.DoubleTensor(grad:size())
  for i = 1, grad:size(1) do
    x[i] = x[i] - eps
    local f1 = f(x)
    x[i] = x[i] + 2.0*eps
    local f2 = f(x)
    x[i] = x[i] - eps
    grad_est[i] = (f2-f1)/(2.0*eps)
  end

  -- computes (symmetric) relative error of gradient
  local diff = torch.norm(grad - grad_est) / torch.norm(grad + grad_est)
  return diff, grad, grad_est
end

function fakedata(n)
    local data = {}
    data.inputs = torch.randn(n, 4)                     -- random standard normal distribution for inputs
    data.targets = torch.rand(n):mul(3):add(1):floor()  -- random integers from {1,2,3}
    return data
end

---------------------------------------------------------
-- generate fake data, then do the gradient check
--
torch.manualSeed(1)
local data = fakedata(5)
local model, criterion = create_model(opt)
local parameters, gradParameters = model:getParameters()

-- returns loss(params)
local f = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  return criterion:forward(model:forward(data.inputs), data.targets)
end
-- returns dloss(params)/dparams
local g = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  gradParameters:zero()

  local outputs = model:forward(data.inputs)
  criterion:forward(outputs, data.targets)
  model:backward(data.inputs, criterion:backward(outputs, data.targets))

  return gradParameters
end

local diff = checkgrad(f, g, parameters)
print(diff, eps)

