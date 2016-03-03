require 'torch'
require 'pl'
require 'trepl'
require 'MyReLU'
require 'MyReQU'
require 'MySigmoid'
require 'MyTanh'
require 'nn'
require 'gnuplot'



opt = lapp[[
   -p,--type               (default float)       float or cuda
      --model              (default MyLinear)    network model
]]

Name = opt.model

-- Generate input X
-- TODO
X = torch.linspace(-10,10,100)
-- Generate ouput using the layer implemented
if opt.model == 'MySigmoid' then
    -- TODO
    Y =  nn.MySigmoid():updateOutput(X)
elseif opt.model == 'MyTanh' then
    -- TODO
    Y =  nn.MyTanh():updateOutput(X)
elseif opt.model == 'MyReLU' then
    --TODO
    Y =  nn.MyReLU():updateOutput(X)
elseif opt.model == 'MyReQU' then
    --TODO
    Y =  nn.MyReQU():updateOutput(X)
end


-- plot and save the figure for Y vs X , Y1 vs X
gnuplot.pngfigure(Name .. '.png')
gnuplot.plot(
   {Name,  X,  Y,  '-'}
   )
gnuplot.xlabel('Input')
gnuplot.ylabel('Output')
gnuplot.plotflush()

