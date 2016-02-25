--[[ A plain implementation of SGD

ARGS:

- `opfunc` : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- `x`      : the initial point
- `config` : a table with configuration parameters for the optimizer
- `config.learningRate`      : learning rate
- `config.learningRateDecay` : learning rate decay
- `config.weightDecay`       : weight decay
- `config.weightDecays`      : vector of individual weight decays
- `config.momentum`          : momentum
- `config.dampening`         : dampening for momentum
- `config.nesterov`          : enables Nesterov momentum
- `config.learningRates`     : vector of individual learning rates
- `state`  : a table describing the state of the optimizer; after each
             call the state is modified
- `state.evalCounter`        : evaluation counter (optional: 0, by default)

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

(Clement Farabet, 2012)
]]
function sgd(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local lrd = config.learningRateDecay or 0
   local wd = config.weightDecay or 0
   local mom = config.momentum or 0
   local damp = config.dampening or mom
   local nesterov = config.nesterov or false
   local lrs = config.learningRates
   local wds = config.weightDecays
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter
   assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)

   -- (2) weight decay with single or individual parameters
   --  = dfdx + wd*x 
   -- wd or wds = lambda
   if opt.optMethod == 'sgd_w' or opt.optMethod == 'sgd_wm' then
      -- TODO ---------------------------------------------




      ----------------------------------------------------
   end

   -- (3) apply momentum
   -- = mom*(1-damp)* (dfdx + wd*x)
   if opt.optMethod == 'sgd_wm' then
      if mom ~= 0 then
         -- TODO ---------------------------------------------





         ----------------------------------------------------
      end
   end

   -- (4) learning rate decay (annealing)
   local clr = lr / (1 + nevals*lrd)
   
   -- (5) parameter update with single or individual learning rates
   if lrs then
      -- TODO ---------------------------------------------




      ----------------------------------------------------
   else
      -- TODO ---------------------------------------------

      ----------------------------------------------------
   end

   -- (6) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization
   return x,{fx}
end
