require 'nn'

local MySigmoid = torch.class('nn.MySigmoid', 'nn.Module')

function MySigmoid:updateOutput(input)

  self.output:resizeAs(input):copy(input)

  -- TODO ----------------------------------
  -- ...something here...
  self.output:mul(-1):exp():add(1):pow(-1)
  ------------------------------------------

  return self.output
end

function MySigmoid:updateGradInput(input, gradOutput)

  self.gradInput:resizeAs(gradOutput):copy(input)

  -- TODO ----------------------------------
  -- ...something here...

  self.gradInput:mul(-1):exp():add(1):pow(-1)
  self.gradInput:addcmul(-1,self.gradInput,self.gradInput):cmul(gradOutput)
  ------------------------------------------

  return self.gradInput
end

