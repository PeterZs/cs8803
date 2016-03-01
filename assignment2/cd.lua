--[[ A plain implementation of Coordinate Descent (CD)

ARGS:

- `opfunc` : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- `x`      : the initial point
- `config` : a table with configuration parameters for the optimizer
- `config.learningRate`      : learning rate
- `config.learningRateDecay` : learning rate decay
- `config.weightDecay`       : weight decay
- `config.weightDecays`      : vector of individual weight decays
- `config.learningRates`     : vector of individual learning rates
- `state`  : a table describing the state of the optimizer; after each
             call the state is modified
- `state.evalCounter`        : evaluation counter (optional: 0, by default)

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

]]

function cd(opfunc, x, run_passed, mean_dfdx, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local lrd = config.learningRateDecay or 0
   local wd = config.weightDecay or 0
   local lrs = config.learningRates
   local wds = config.weightDecays
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)

   -- (2) learning rate decay (annealing)
   local clr = lr / (1 + nevals*lrd)

   -- TODO ---------------------------------------------
   -- (3) select an index of the weight vect x
   -- (4) Make dfdx into zero except an element with the selcted index
   -- dfdx
   local index = torch.random(1,dfdx:size()[1])
   local indexVector = torch.zeros(dfdx:size()[1])
   indexVector[index] = 1
   dfdx:cmul(indexVector)

   ----------------------------------------------------

   -- (5) parameter update with single learning rates
   -- TODO ---------------------------------------------
   -- x
   x:add(-clr, dfdx)

   ----------------------------------------------------

   -- (6) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization
   return x,{fx}
end
