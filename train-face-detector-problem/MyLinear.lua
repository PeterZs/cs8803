-- In this you will be implementing the MyLinear module
-- This module takes in: 
-- "inputSize" inputs 
-- "outputSize" outputs
-- The output is calculated as output = weight*input+bias
-- You may refer to the Liner.lua file in the official torch package

local MyLinear, parent = torch.class('nn.MyLinear', 'nn.Module')

-- Constructor to create parameters and initilize them
function MyLinear:__init(inputSize, outputSize)
   parent.__init(self)
   -- Instantiate all the varaibles that are required
   -- For this layer we require weight, bias, gradWeight, gradBias 
   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)

   -- Calling reset
   self:reset()
end


-- This method defines how the trainable parameters are reset, 
-- i.e. initialized before training
function MyLinear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end

   -- Initiliaze the parameters randomly
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)

   return self
end

-- This function computes the output using the current parameter
-- set of the class and input.
-- For MyLinear layer the output is calculated as weight*input+bias
function MyLinear:updateOutput(input)
   -- Your code should handle 2 cases.
   -- 1. when input is only of one dimenstion i.e. the batch size 
   --    is only 1
   -- 2. When the input is 2D, i.e. the batch size is more than 1    
   if input:dim() == 1 then

      self.output:resize(self.bias:size(1))
      -- TODO ---------------------------------------------
      -- self.output


      -----------------------------------------------------
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()

      self.output:resize(nframe, self.bias:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end

      -- TODO ---------------------------------------------
      -- self.output
      
      -----------------------------------------------------

   else
      error('input must be vector or matrix')
   end

   return self.output
end

-- This function computes gradInput given the input to the layer and
-- the gradOutput i.e the gradtient form the previous layer
-- Refer to the document on how gradInput is computed
function MyLinear:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         -- TODO ---------------------------------------------
         -- self.gradInput

         -----------------------------------------------------

      elseif input:dim() == 2 then
         -- TODO ---------------------------------------------
         -- self.gradInput

         -----------------------------------------------------
      end

      return self.gradInput
   end
end

-- Accumulate the gradient of each of the trainable parameters
function MyLinear:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      -- TODO ---------------------------------------------
      -- self.gradWeight
      -- self.gradBias

      -----------------------------------------------------
   elseif input:dim() == 2 then
      -- TODO ---------------------------------------------
      -- self.gradWeight
      -- self.gradBias

      -----------------------------------------------------
   end
end

-- we do not need to accumulate parameters when sharing
MyLinear.sharedAccUpdateGradParameters = MyLinear.accUpdateGradParameters


function MyLinear:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
