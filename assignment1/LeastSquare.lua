require 'torch'
require 'gnuplot'

X = torch.linspace(0,100,10)
Y = X*0.5 + 0.2 + torch.rand(10) *10
theta = torch.rand(2)

-- Fitting a line
function f(theta)
    return X*theta[1]+theta[2]
end

function J(theta,lamda)
    -- return the cost function
	return torch.dot(Y-f(theta),Y-f(theta)) + lamda * torch.dot(theta,theta)
end

function dJ(theta,lamda)
    -- return the gradient of cost function wrt theta
	return torch.Tensor({
		-2 * torch.dot(X,Y-f(theta)) + 2 * lamda * theta[1],
		-2 * torch.dot(torch.ones(#X),Y-f(theta)) + 2 * lamda * theta[2],
		}) 
end


lr = 0.000001
lamda = 0.1
J_hist = {}
for i=1,200 do
	theta = theta - dJ(theta,lamda):mul(lr)
	J_hist[i] = J(theta,lamda)
	-- we print the value of the objective function at each iteration
	-- print(string.format('at iter %d J(x) = %f', i, J(theta,lamda)))
end

Y1 = f(theta)

-- plot and save the figure for Y vs X , Y1 vs X
gnuplot.pngfigure('plot.png')
gnuplot.plot(
   {'Y',  X,  Y,  '-'},
   {'Y pred',  X,  Y1,  '-'}
   )
gnuplot.xlabel('Input')
gnuplot.ylabel('Output')
gnuplot.plotflush()
-- plot and save the figure of J vs iteration 
-- include the two plots in the 
gnuplot.pngfigure('cost_function.png')
gnuplot.plot({ torch.range(1, #J_hist), torch.Tensor(J_hist), '-'})
gnuplot.xlabel('No. of iterations')
gnuplot.ylabel('Cost function')
gnuplot.plotflush()

