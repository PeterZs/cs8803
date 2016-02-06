-- In this you will be implementing the MyLinearSigmoid module
-- This module takes in: 
-- "inputSize" inputs 
-- "outputSize" outputs
-- The output is calculated as output = sigmoid(weight*input+bias)

local MyLinearSigmoid, parent = torch.class('nn.MyLinearSigmoid', 'nn.Module')

-- Constructor to create parameters and initilize them
function MyLinearSigmoid:__init(inputSize, outputSize)
   parent.__init(self)
   -- Instantiate all the varaibles that are required
   -- For this layer we require weight, bias, gradWeight, gradBias,
   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)

   -- z1 stores the output of the Linear layer
   -- a1 stores the output of the sigmoid layer
   -- gradA1 is the gradInput of the sigmoid layer
   self.z1 = torch.Tensor()
   self.a1 = torch.Tensor()
   self.gradA1 = torch.Tensor()

   -- Calling reset
   self:reset()
end

-- This method defines how the trainable parameters are reset, 
-- i.e. initialized before training
function MyLinearSigmoid:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   -- initialize weight and bias with random values 
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)

   return self 
end

-- This function computes the output using the current parameter
-- set of the class and input.
-- For MyLinearSigmoid layer the output is calculated as 
-- sigmoid(weight*input+bias)
function MyLinearSigmoid:updateOutput(input)
   -- Your code should handle 2 cases.
   -- 1. when input is only of one dimenstion i.e. the batch size 
   --    is only 1
   -- 2. When the input is 2D, i.e. the batch size is more than 1    

   -- Calcualting z1 i.e the output of the Linear part of the layer
   if input:dim() == 1 then
      self.z1:resize(self.bias:size(1))
      -- TODO ---------------------------------------------
	  self.z1:copy(self.bias)
	  self.z1:addmv(self.weight, input)


      --------------------------------------------------------
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.z1:nElement()
      self.z1:resize(nframe, self.bias:size(1))
      if self.z1:nElement() ~= nElement then
         self.z1:zero()
      end
      -- TODO ---------------------------------------------
	  self.z1:addmm(0, self.z1, 1, input, self.weight:t())
	  self.z1:addr(torch.ones(nframe), self.bias)


      --------------------------------------------------------
   else
      error('input must be vector or matrix')
   end

   -- calcualting a1 i.e the output of the sigmoid(z1)
   -- self.a1 = sigmoid(self.z1)
	self.a1 = torch.pow(torch.add(torch.exp(-self.z1),1),-1)

   --
   self.output:resizeAs(self.a1):copy(self.a1)

   return self.output
end

-- This function computes gradInput given the input to the layer and
-- the gradOutput i.e the gradtient form the previous layer
-- Refer to the document on how gradInput is computed
function MyLinearSigmoid:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end

      if input:dim() == 1 then

         -- for activation module
         -- TODO ---------------------------------------------
         -- self.gradA1
		self.gradA1 = torch.cmul(gradOutput,-torch.cmul(self.a1,torch.add(self.a1,-1)))
         -----------------------------------------------------

         -- for linear 1 module
         -- TODO ---------------------------------------------
         -- self.gradInput
		 self.gradInput:addmv(self.weight:t(), self.gradA1)

         -----------------------------------------------------

      elseif input:dim() == 2 then
         -- for activation module
         -- TODO ---------------------------------------------
         -- self.gradA1
		self.gradA1 = torch.cmul(gradOutput,-torch.cmul(self.a1,torch.add(self.a1,-1)))
         -----------------------------------------------------

         -- for linear 1 module
         -- TODO ---------------------------------------------
         -- self.gradInput
		 self.gradInput:addmm(self.gradA1, self.weight)

         -----------------------------------------------------
      end

      return self.gradInput
   end
end


-- Accumulate the gradient of each of the trainable parameters
function MyLinearSigmoid:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      -- TODO ---------------------------------------------
      -- self.gradWeight
      -- self.gradBias
	  self.gradWeight:addr(scale, self.gradA1, input)
	  self.gradBias:add(scale, self.gradA1)

      -----------------------------------------------------
   elseif input:dim() == 2 then
      -- TODO ---------------------------------------------
      -- self.gradWeight
      -- self.gradBias
      local nframe = input:size(1)
	  self.gradWeight:addmm(scale, self.gradA1:t(), input)
	  self.gradBias:addmv(scale, self.gradA1:t(), torch.ones(nframe))

      -----------------------------------------------------
   end
end

-- we do not need to accumulate parameters when sharing
MyLinearSigmoid.sharedAccUpdateGradParameters = MyLinearSigmoid.accUpdateGradParameters


function MyLinearSigmoid:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end

-- return the output of sigmoid input
function sigmoid(input)

   -- TODO ---------------------------------------------
	return torch.pow(torch.add(torch.exp(-input),1),-1)

   -----------------------------------------------------
end

