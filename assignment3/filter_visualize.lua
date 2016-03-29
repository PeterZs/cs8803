require 'nn'
require 'optim'
require 'image'
--- Sample code to visulaize the filter weigths.
--  Save your network once you have trained and visualzie the weights.
model = nn.Sequential()
model = torch.load('results/model.net')

cnn = model:get(1)
win_w1 = image.display{
   image=cnn:get(1).weight, zoom=4, nrow=4,
   min=-1, max=1,
   win=win_w1, legend='stage 1: weights', padding=1
}
win_w2 = image.display{
   --image=cnn:get(4).weight[{{},{1},{},{}}], zoom=4, nrow=4,
   image=cnn:get(4).weight, zoom=4, nrow=4,
   min=-1, max=1,
   win=win_w2, legend='stage 2: weights', padding=1
}

image.save('stage_1_weights.png', cnn:get(1).weight)
image.save('stage_2_weights.png', cnn:get(4).weight)
