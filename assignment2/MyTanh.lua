require 'nn'

local MyTanh = torch.class('nn.MyTanh', 'nn.Module')

function MyTanh:updateOutput(input)

  self.output:resizeAs(input):copy(input)

  -- TODO ----------------------------------
  -- ...something here...
  self.output:tanh()
  ------------------------------------------

  return self.output
end

function MyTanh:updateGradInput(input, gradOutput)

  self.gradInput:resizeAs(gradOutput):copy(gradOutput)

  -- TODO ----------------------------------
  -- ...something here...
  local h = torch.tanh(input)
  self.gradInput:cmul(h,h):mul(-1):add(1):cmul(gradOutput)

  ------------------------------------------

  return self.gradInput
end

