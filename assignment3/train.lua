----------------------------------------------------------------------
-- This script demonstrates how to define a training procedure,
-- irrespective of the model/loss functions chosen.
--
-- It shows how to:
--   + construct mini-batches on the fly
--   + define a closure to estimate (a noisy) loss
--     function, as well as its derivatives wrt the parameters of the
--     model to be trained
--   + optimize the function, according to several optmization
--     methods: SGD, L-BFGS.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'
require 'image'

----------------------------------------------------------------------
-- Model + Loss:
local t = require 'model'
local model = t.model
local fwmodel = t.model
local loss = t.loss
local model_name = t.model_name

----------------------------------------------------------------------
-- Save light network tools:
function nilling(module)
   module.gradBias   = nil
   if module.finput then module.finput = torch.Tensor() end
   module.gradWeight = nil
   module.output     = torch.Tensor()
   if module.fgradInput then module.fgradInput = torch.Tensor() end
   module.gradInput  = nil
end

function netLighter(network)
   nilling(network)
   if network.modules then
      for _,a in ipairs(network.modules) do
         netLighter(a)
      end
   end
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> defining some tools')

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)

-- Log results to files
local trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> flattening model parameters')

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
local w,dE_dw = model:getParameters()

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> configuring optimizer')

local optimState = {
   learningRate = opt.learningRate,
   momentum = opt.momentum,
   weightDecay = opt.weightDecay,
   learningRateDecay = opt.learningRateDecay
}
local function deepCopy(object)
    local lookup_table = {}
    local function _copy(object)
        if type(object) ~= "table" then
            return object
        elseif lookup_table[object] then
            return lookup_table[object]
        end
        local new_table = {}
        lookup_table[object] = new_table
        for index, value in pairs(object) do
            new_table[_copy(index)] = _copy(value)
        end
        return setmetatable(new_table, getmetatable(object))
    end
    return _copy(object)
end
if opt.model == "CNN_FINETUNE" then
    -- TODO ---
    -- The below code assigns learning rate which are 1/10th of the
    -- base learning rate for all the layers except the last one.
    -- You may modify this and freeze all the layers except the last
    optimState1 = {}
    for i=1,#model.modules do
        optimState1[i] = deepCopy(optimState)
        if i ~= #model.modules then 
            optimState1[i].learningRate = optimState.learningRate/10
        end
    end
end

local height = 224
local width = 224
----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> allocating minibatch memory')
local x = torch.Tensor(opt.batchSize,trainData.data:size(2), 
         height, width) --faces data
local yt = torch.Tensor(opt.batchSize)
if opt.type == 'cuda' then 
   x = x:cuda()
   yt = yt:cuda()
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> defining training procedure')

local epoch
local function data_augmentation(inputs)
    if opt.dataAugment == 'true' then
        -- TODO---
        -- DATA augmentation, mirroring and cropping
		local x = torch.rand(1):mul(256-width):floor()
		local y = torch.rand(1):mul(256-width):floor()
		local outputs = image.crop(inputs,x,y,x+width,y+height)
		if torch.rand() > 0.5 then
			outputs = image.hflip(outputs)
		end
		if torch.rand() > 0.5 then
			outputs = image.vflip(outputs)
		end
		return outputs
    else
        local outputs = image.crop(inputs,"c",height,width)
        return outputs
    end
end

local function train(trainData)

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()
   local run_passed = 0
   local mean_dfdx = torch.Tensor():typeAs(w):resizeAs(w):zero()

   -- shuffle at each epoch
   local shuffle = torch.randperm(trainData:size())

   -- do one epoch
   print(sys.COLORS.green .. '==> doing epoch on training data:') 
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,trainData:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, trainData:size())
      collectgarbage()

      -- batch fits?
      if (t + opt.batchSize - 1) > trainData:size() then
         break
      end

      -- create mini batch
      local idx = 1
      for i = t,t+opt.batchSize-1 do
         x[idx] = data_augmentation(trainData.data[shuffle[i]])
         yt[idx] = trainData.labels[shuffle[i]]
         idx = idx + 1
      end
      if opt.model == "CNN_FINETUNE" then
          --TODO ----
          --Write the code necessary to fine tune the 
          --model.
      else
         -- create closure to evaluate f(X) and df/dX
         local eval_E = function(w)
            -- reset gradients
            dE_dw:zero()

            -- evaluate function for complete mini batch
            local y = model:forward(x)
            local E = loss:forward(y,yt)

            -- estimate df/dW
            local dE_dy = loss:backward(y,yt)   
            model:backward(x,dE_dy)

            -- update confusion
            for i = 1,opt.batchSize do
               confusion:add(y[i],yt[i])
            end

            -- return f and df/dX
            return E,dE_dw
         end

         -- optimize on current mini-batch
         if opt.optMethod == 'sgd' then
            optim.sgd(eval_E, w, optimState)
         elseif opt.optMethod == 'asgd' then
            run_passed = run_passed + 1
            mean_dfdx  = asgd(eval_E, w, run_passed, mean_dfdx, optimState)
         end                                
     end
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   --local filename = paths.concat(opt.save, model_name)
   --os.execute('mkdir -p ' .. sys.dirname(filename))
   --print('==> saving model to '..filename)
   --model1 = model:clone()
   --netLighter(model1)
   --torch.save(filename, model1)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end

-- Export:
return train

