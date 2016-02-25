require 'nn'

local MyReQU = torch.class('nn.MyReQU', 'nn.Module')

function MyReQU:updateOutput(input)

  self.output:resizeAs(input):copy(input)

  -- TODO ----------------------------------
  -- ...something here...


  ------------------------------------------

  return self.output
end

function MyReQU:updateGradInput(input, gradOutput)

  self.gradInput:resizeAs(gradOutput):copy(gradOutput)

  -- TODO ----------------------------------
  -- ...something here...



  ------------------------------------------

  return self.gradInput
end

