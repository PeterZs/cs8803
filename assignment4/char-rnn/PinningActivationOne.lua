require 'nn'

local PinningActivationOne = torch.class('nn.PinningActivationOne', 'nn.Module')

function PinningActivationOne:updateOutput(input)

  -- TODO ---------------------------
  self.output = torch.ones(input:size())
  --self.output = input:clone()

  return self.output
end

function PinningActivationOne:updateGradInput(input, gradOutput)

  -- TODO ---------------------------
  --self.gradInput = gradOutput:clone()
  self.gradInput = torch.zeros(gradOutput:size())

  return self.gradInput
end

