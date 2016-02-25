require 'nn'

local MyReLU = torch.class('nn.MyReLU', 'nn.Module')

function MyReLU:updateOutput(input)

  self.output:resizeAs(input):copy(input)

  -- TODO ----------------------------------
  -- ...something here...

  ------------------------------------------

  return self.output
end

function MyReLU:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)

  -- TODO ----------------------------------
  -- ...something here...

  ------------------------------------------

  return self.gradInput
end

