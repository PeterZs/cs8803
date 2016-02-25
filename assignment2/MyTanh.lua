require 'nn'

local MyTanh = torch.class('nn.MyTanh', 'nn.Module')

function MyTanh:updateOutput(input)

  self.output:resizeAs(input):copy(input)

  -- TODO ----------------------------------
  -- ...something here...

  ------------------------------------------

  return self.output
end

function MyTanh:updateGradInput(input, gradOutput)

  self.gradInput:resizeAs(gradOutput):copy(gradOutput)

  -- TODO ----------------------------------
  -- ...something here...


  ------------------------------------------

  return self.gradInput
end

