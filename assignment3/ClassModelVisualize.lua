require 'nn'
require 'optim'
require 'image'
require 'loadcaffe'
require 'xlua'

local vgg_net = loadcaffe.load('models/VGG_CNN_M_deploy.prototxt', 'models/VGG_CNN_M.caffemodel', 'nn')


local function clean(input_model)
    model = input_model:clone()
    for i=1,#model.modules do
        if tostring(model.modules[i]) == 'nn.SoftMax' then
            table.remove(model.modules, i)    
        end
        if tostring(model.modules[i]) == 'nn.Dropout' then
            model.modules[i].train = false    
            model.modules[i].updateGradInput = function(input, gradOutput)
                return model.modules[i].gradInput
            end
        end
    end
    return model
end
-- remove the softmax layer
local vgg_net_nosoftmax = clean(vgg_net)


local model = nn.Sequential()
local H = 224
local W = 224
local channel = 3
local img_mean_name = 'ilsvrc_2012_mean.t7'
local img_mean_url = 'https://www.dropbox.com/s/p33rheie3xjx6eu/'..img_mean_name
if not paths.filep(img_mean_name) then os.execute('wget '..img_mean_url) end

local img_mean = torch.load('ilsvrc_2012_mean.t7').img_mean:transpose(3,1):float()

img_mean = image.scale(img_mean,H,W)
local zero_image = torch.zeros(channel,H,W)
--model:add(nn.ImageLayer(zero_image))
model:add(vgg_net_nosoftmax)
print(model)




local loss = nn.ClassNLLCriterion()


config = config or {learningRate = 1e4,
                      weightDecay = 1e-5,
                      momentum = 0.9,
                      learningRateDecay = 5e-7
                      }
t = 131
target = torch.zeros(1)
target[1] = t

--model.modules[1]:reset(zero_image:clone())
print('Generating class model image of class ' .. t)
current_image = zero_image
win_w1 = image.display{
   image=current_image, zoom=1, nrow=1,
   min=-1, max=1,
   win=win_w1, legend='Initial Image', padding=1
}
iter = 25
for e = 1,iter do
    xlua.progress(e, iter)
    -- TODO ---
    -- Optimize the image to maximize the score for target.
	local y = model:forward(current_image)
	local E = loss:forward(y,target)
	local dE_dy = loss:backward(y,target)
	model:backward(current_image,dE_dy)
	current_image:add(model.modules[1].gradInput)
    
     win_w2 = image.display{
        image=current_image, zoom=1, nrow=1,
        min=-1, max=1,
        win=win_w2, legend='Image after iter '..e, padding=1
     }
end

-- Adding the mean image ---
current_image:add(1,img_mean:double())
current_image = current_image:index(1,torch.LongTensor{3,2,1})
print(current_image:min())
print(current_image:max())
current_image = current_image:add(-current_image:min())
current_image = current_image:div(current_image:max())
-- save class model image
image.save('./' .. t .. '.png', current_image)
