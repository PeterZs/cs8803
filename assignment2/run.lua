----------------------------------------------------------------------
-- Train a ConvNet on faces.
--
-- original: Clement Farabet
-- new version by: E. Culurciello 
-- Mon Oct 14 14:58:50 EDT 2013
----------------------------------------------------------------------

require 'pl'
require 'trepl'
require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> processing options')

opt = lapp[[
   -r,--learningRate       (default 1e-3)        learning rate
   -d,--learningRateDecay  (default 1e-7)        learning rate decay (in # samples)
   -w,--weightDecay        (default 1e-5)        L2 penalty on the weights
   -m,--momentum           (default 0.1)         momentum
   -d,--dropout            (default 0.5)         dropout amount
   -b,--batchSize          (default 128)         batch size
   -t,--threads            (default 8)           number of threads
   -p,--type               (default float)       float or cuda
   -i,--devid              (default 1)           device ID (if using CUDA)
   -s,--size               (default small)       dataset: small or full or extra
   -o,--save               (default results)     save directory
      --patches            (default all)         percentage of samples to use for testing'
      --visualize          (default false)       visualize dataset
      --model              (default CNN)         network model
      --optMethod          (default optim_sgd)   optimization method
      --plot               (default false)       plot the training and test accuracies
]]

-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- type:
if opt.type == 'cuda' then
   print(sys.COLORS.red ..  '==> switching to CUDA')
   require 'cunn'
   cutorch.setDevice(opt.devid)
   print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> load modules')

local data  = require 'data'
local train = require 'train'
local test  = require 'test'

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> training!')

for i=1,100 do
   train(data.trainData)
   test(data.testData)
end

