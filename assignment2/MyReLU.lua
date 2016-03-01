require 'nn'

local MyReLU = torch.class('nn.MyReLU', 'nn.Module')

function MyReLU:updateOutput(input)

  self.output:resizeAs(input):copy(input)

  -- TODO ----------------------------------
  -- ...something here...
  self.output:cmul(torch.gt(self.output,0):typeAs(self.output))
  ------------------------------------------

  return self.output
end

function MyReLU:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)

  -- TODO ----------------------------------
  -- ...something here...
  self.gradInput:cmul(torch.gt(input,0):typeAs(self.output))
  ------------------------------------------

  return self.gradInput
end

