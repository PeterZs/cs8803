-- In this you will be implementing the MyLinearSigmoidLinear module
-- This module takes in: 
-- "inputSize" inputs 
-- "outputSize" outputs
-- The output is calculated as output = sigmoid(weight*input+bias)

local MyLinearSigmoidLinear, parent = torch.class('nn.MyLinearSigmoidLinear', 'nn.Module')

-- Constructor to create parameters and initilize them
function MyLinearSigmoidLinear:__init(inputSize, hiddenSize, outputSize)
   parent.__init(self)

   -- Instantiate all the varaibles that are required
   -- For this layer we require weight1, bias1, gradWeight1, gradBias1,
   -- weight2, bias2, gradWeight2, gradBias2,
   self.weight1 = torch.Tensor(hiddenSize, inputSize)
   self.bias1 = torch.Tensor(hiddenSize)
   self.gradWeight1 = torch.Tensor(hiddenSize, inputSize)
   self.gradBias1 = torch.Tensor(hiddenSize)

   self.weight2 = torch.Tensor(outputSize, hiddenSize)
   self.bias2 = torch.Tensor(outputSize)
   self.gradWeight2 = torch.Tensor(outputSize, hiddenSize)
   self.gradBias2 = torch.Tensor(outputSize)


   -- z1 stores the output of the first Linear layer
   -- z2 stores the output of the second Linear layer
   -- a1 stores the output of the first sigmoid layer
   -- a2 stores the output of the second sigmoid layer
   -- gradA1 is the gradInput of the sigmoid layer
   -- gradA2 is the gradInput of the second Linear layer
   self.z1 = torch.Tensor()
   self.z2 = torch.Tensor()
   self.a1 = torch.Tensor()
   self.a2 = torch.Tensor()
   self.gradA1 = torch.Tensor()
   self.gradA2 = torch.Tensor()

   -- Calling reset
   self:reset()
end

-- This method defines how the trainable parameters are reset, 
-- i.e. initialized before training
function MyLinearSigmoidLinear:reset()
   -- initialize weight and bias with random values 
   stdv = 1./math.sqrt(self.weight1:size(2))
   self.weight1:uniform(-stdv, stdv)
   self.bias1:uniform(-stdv, stdv)

   stdv = 1./math.sqrt(self.weight2:size(2))
   self.weight2:uniform(-stdv, stdv)
   self.bias2:uniform(-stdv, stdv)

   return self 
end

-- This function computes the output using the current parameter
-- set of the class and input.
-- For MyLinearSigmoid layer the output is calculated as 
-- weight2*sigmoid(weight1*input+bias1)+bias2
function MyLinearSigmoidLinear:updateOutput(input)
   -- Your code should handle 2 cases.
   -- 1. when input is only of one dimenstion i.e. the batch size 
   --    is only 1
   -- 2. When the input is 2D, i.e. the batch size is more than 1    

   -- Calcualting z1 i.e the output of the first Linear part of the layer
   if input:dim() == 1 then
      self.z1:resize(self.bias1:size(1))
      -- TODO ---------------------------------------------
	  self.z1:copy(self.bias1)
	  self.z1:addmv(self.weight1, input)


      -----------------------------------------------------
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.z1:nElement()
      self.z1:resize(nframe, self.bias1:size(1))
      if self.z1:nElement() ~= nElement then
         self.z1:zero()
      end
      -- TODO ---------------------------------------------
	  self.z1:addmm(0, self.z1, 1, input, self.weight1:t())
	  self.z1:addr(torch.ones(nframe), self.bias1)

      -----------------------------------------------------
   else
      error('input must be vector or matrix')
   end

   -- calcualting a1 i.e the output of the sigmoid(z1)
   self.a1 = sigmoid(self.z1)

   -- Calcualting z2 i.e the output of the second Linear part of the layer
   if self.a1:dim() == 1 then
      -- TODO ---------------------------------------------
      self.z2:resize(self.bias2:size(1))
	  self.z2:copy(self.bias2)
	  self.z2:addmv(self.weight2, self.a1)

      -----------------------------------------------------
   elseif self.a1:dim() == 2 then
      local nframe = self.a1:size(1)
      local nElement = self.z2:nElement()
      self.z2:resize(nframe, self.bias2:size(1))
      if self.z2:nElement() ~= nElement then
         self.z2:zero()
      end
      -- TODO ---------------------------------------------
	  self.z2:addmm(0, self.z2, 1, self.a1, self.weight2:t())
	  self.z2:addr(torch.ones(nframe), self.bias2)
      -----------------------------------------------------
   else
      error('input must be vector or matrix')
   end

   --
   self.output:resizeAs(self.z2):copy(self.z2)



   return self.output
end

-- This function computes gradInput given the input to the layer and
-- the gradOutput i.e the gradtient form the previous layer
-- Refer to the document on how gradInput is computed
function MyLinearSigmoidLinear:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      local nElement2 = self.gradA2:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      self.gradA2:resizeAs(self.a1)
      if self.gradA2:nElement() ~= nElement2 then
         self.gradA2:zero()
      end

      if input:dim() == 1 then

         -- for the second linear module
         -- TODO ---------------------------------------------
         -- self.gradA2
		 self.gradA2:resize(self.a1:size())
		 self.gradA2:addmv(self.weight2:t(), gradOutput)
         -----------------------------------------------------

         -- for the first activation module
         -- TODO ---------------------------------------------
		 self.gradA1:cmul(self.gradA2,-torch.cmul(self.a1,torch.add(self.a1,-1)))
         -----------------------------------------------------

         -- for the first linear module
         -- TODO ---------------------------------------------
		 self.gradInput:addmv(self.weight1:t(), self.gradA1)

         -----------------------------------------------------

      elseif input:dim() == 2 then
         -- for the second linear module
         -- TODO ---------------------------------------------
		 self.gradA2:resize(self.a1:size())
		 self.gradA2:addmm(gradOutput, self.weight2)

         -----------------------------------------------------

         -- for the first activation module
         -- TODO ---------------------------------------------
		 self.gradA1:cmul(self.gradA2,-torch.cmul(self.a1,torch.add(self.a1,-1)))
         -----------------------------------------------------

         -- for the first linear module
         -- TODO ---------------------------------------------
		 self.gradInput:addmm(self.gradA1, self.weight1)

         -----------------------------------------------------
      end

      return self.gradInput
   end
end

function MyLinearSigmoidLinear:parameters()
   return {self.weight1, self.weight2, self.bias1, self.bias2}, {self.gradWeight1, self.gradWeight2, self.gradBias1, self.gradBias2}
end

function MyLinearSigmoidLinear:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      -- TODO ---------------------------------------------
	  self.gradWeight2:addr(scale, gradOutput, self.a1)
	  self.gradBias2:add(scale, gradOutput)
	  self.gradWeight1:addr(scale, self.gradA1, input)
	  self.gradBias1:add(scale, self.gradA1)


      -----------------------------------------------------
   elseif input:dim() == 2 then
      -- TODO ---------------------------------------------
	  self.gradWeight2:addmm(0,self.gradWeight2,scale, gradOutput:t(), self.a1)
	  self.gradBias2:addmv(0,self.gradBias2,scale, gradOutput:t(), torch.ones(self.z2:size(1)))
	  self.gradWeight1:addmm(0,self.gradWeight1,scale, self.gradA1:t(), input)
	  self.gradBias1:addmv(0,self.gradBias1,scale, self.gradA1:t(), torch.ones(self.a1:size(1)))

      -----------------------------------------------------
   end
end

-- we do not need to accumulate parameters when sharing
MyLinearSigmoidLinear.sharedAccUpdateGradParameters = MyLinearSigmoidLinear.accUpdateGradParameters


function MyLinearSigmoidLinear:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight1:size(2), self.weight2:size(1))
end


function sigmoid(input)

   -- TODO ---------------------------------------------
	return torch.pow(torch.add(torch.exp(-input),1),-1)

   -----------------------------------------------------
end

