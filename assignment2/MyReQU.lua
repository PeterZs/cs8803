require 'nn'

local MyReQU = torch.class('nn.MyReQU', 'nn.Module')

function MyReQU:updateOutput(input)

  self.output:resizeAs(input):copy(input)

  -- TODO ----------------------------------
  -- ...something here...
  self.output:cmul(torch.gt(self.output,0):typeAs(self.output)):cmul(self.output)
  ------------------------------------------

  return self.output
end

function MyReQU:updateGradInput(input, gradOutput)

  self.gradInput:resizeAs(gradOutput):copy(gradOutput)

  -- TODO ----------------------------------
  -- ...something here...
  self.gradInput:cmul(torch.gt(input,0):typeAs(self.output)):cmul(input):mul(2)
  ------------------------------------------

  return self.gradInput
end

