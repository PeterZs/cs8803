----------------------------------------------------------------------
-- This script demonstrates how to load the Face Detector 
-- training data, and pre-process it to facilitate learning.
--
-- It's a good idea to run this script with the interactive mode:
-- $ torch -i 1_data.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to analyze/visualize the data you've just loaded.
--
-- Clement Farabet, Eugenio Culurciello
-- Mon Oct 14 14:58:50 EDT 2013
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nnx'      -- provides a normalization operator

local opt = opt or {
   visualize = true,
   size = 'small',
   patches='all'
}


print(sys.COLORS.red ..  '==> loading dataset')
-- see if the file exists
local function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

-- get all lines from a file, returns an empty 
-- list/table if the file does not exist
local function lines_from(file)
  if not file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do 
    lines[#lines + 1] = line
  end
  return lines
end



-- We load the dataset from disk
-- data bg: from 0 to 28033; face: from 28034 to 41266
local numImages = 1000 
local imagesAll = torch.Tensor(numImages,3,256,256)
local labelsAll = torch.Tensor(numImages)

-- classes: GLOBAL var!
classes = {'Detailed','Pastel','Melancholy','Noir','HDR','Vintage','Long Exposure','Horror','Sunny','Bright','Hazy','Bokeh','Serene','Texture','Ethereal','Macro','Depth of Field','Geometric Composition','Minimal','Romantic'}
if paths.filep('traindata.data') then
	local result = torch.load('traindata.data')
	trainData = result.trainData
	testData = result.testData
	return result
end
-- load backgrounds:
local file_name = 'flickr_style/train.txt'
local lines = lines_from(file_name)
local trsize = #lines 
for k,v in pairs(lines) do
  local image_name,class = v:match("([^ ]+) ([^ ]+)")
	local im = image.scale(image.load(image_name),256,256)
	if im:size()[1] == 1 then
		imagesAll[k][1] = im
		imagesAll[k][2] = im
		imagesAll[k][3] = im
	else
  		imagesAll[k] = im
	end
  labelsAll[k] = tonumber(class)+1
end

local file_name = 'flickr_style/test.txt'
local lines = lines_from(file_name)
local tesize = #lines 
for k,v in pairs(lines) do
  local image_name,class = v:match("([^ ]+) ([^ ]+)")
	local im = image.scale(image.load(image_name),256,256)
	if im:size()[1] == 1 then
		imagesAll[trsize+k][1] = im
		imagesAll[trsize+k][2] = im
		imagesAll[trsize+k][3] = im
	else
  		imagesAll[trsize+k] = im
	end
  labelsAll[trsize+k] = tonumber(class)+1
end

-- shuffle dataset: get shuffled indices in this variable:
print(sys.COLORS.red ..  '==> shuffle')
local labelsShuffle = torch.randperm((#labelsAll)[1])


-- create train set:
trainData = {
   data = torch.Tensor(trsize, 3, 256, 256),
   labels = torch.Tensor(trsize),
   size = function() return trsize end
}
--create test set:
testData = {
      data = torch.Tensor(tesize, 3, 256, 256),
      labels = torch.Tensor(tesize),
      size = function() return tesize end
   }

for i=1,trsize do
   trainData.data[i] = imagesAll[labelsShuffle[i]]:clone()
   trainData.labels[i] = labelsAll[labelsShuffle[i]]
end
for i=trsize+1,tesize+trsize do
   testData.data[i-trsize] = imagesAll[labelsShuffle[i]]:clone()
   testData.labels[i-trsize] = labelsAll[labelsShuffle[i]]
end

-- remove from memory temp image files:
 imagesAll = nil
 labelsAll = nil


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> preprocessing data')
if opt.model == "CNN_FINETUNE" or opt.model == "CNN_FINETUNE_1" then
   print("Subtracting by mean") 
   --local R = trainData[{ {},1,{},{} }]
   --local B = trainData[{ {},3,{},{} }]
   --trainData[{ {},1,{},{} }] = B
   --trainData[{ {},3,{},{} }] = R
   --local R = testData[{ {},1,{},{} }]
   --local B = testData[{ {},3,{},{} }]
   --testData[{ {},1,{},{} }] = B
   --testData[{ {},3,{},{} }] = R
   --local mean = { 0.48462227599918, 0.45624044862054, 0.40588363755159}
   --local std = {0.22889466674951, 0.22446679341259, 0.22495548344775}
   --for i=1,3  do 
   --   trainData.data[{ {},i,{},{} }]:add(-mean[i]):div(std[i])
   --   testData.data[{ {},i,{},{} }]:add(-mean[i]):div(std[i])
   --end
   --
   local img_mean_name = 'ilsvrc_2012_mean.t7'
   local img_mean_url = 'https://www.dropbox.com/s/p33rheie3xjx6eu/'..img_mean_name
   if not paths.filep(img_mean_name) then os.execute('wget '..img_mean_url) end
   local img_mean = torch.load('ilsvrc_2012_mean.t7').img_mean:transpose(3,1):float()
   print(img_mean:size())
   print(trainData.data:size())
   trainData.data:mul(255)
   testData.data:mul(255)
   trainData.data = trainData.data:index(2,torch.LongTensor{3,2,1})
   testData.data = testData.data:index(2,torch.LongTensor{3,2,1})
   trainData.data:add(-1, torch.repeatTensor(img_mean,trainData.data:size(1),1,1,1))
   testData.data:add(-1, torch.repeatTensor(img_mean,testData.data:size(1),1,1,1))
   print(trainData.data:size())
else
   -- preprocess trainSet
   normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
   for i = 1,trainData:size() do
      -- rgb -> yuv
      local rgb = trainData.data[i]
      local yuv = image.rgb2yuv(rgb)
      -- normalize y locally:
      yuv[1] = normalization(yuv[{{1}}])
      trainData.data[i] = yuv
   end
   -- normalize u globally:
   mean_u = trainData.data[{ {},2,{},{} }]:mean()
   std_u = trainData.data[{ {},2,{},{} }]:std()
   trainData.data[{ {},2,{},{} }]:add(-mean_u)
   trainData.data[{ {},2,{},{} }]:div(-std_u)
   -- normalize v globally:
   mean_v = trainData.data[{ {},3,{},{} }]:mean()
   std_v = trainData.data[{ {},3,{},{} }]:std()
   trainData.data[{ {},3,{},{} }]:add(-mean_v)
   trainData.data[{ {},3,{},{} }]:div(-std_v)
   
   -- preprocess testSet
   for i = 1,testData:size() do
      -- rgb -> yuv
      local rgb = testData.data[i]
      local yuv = image.rgb2yuv(rgb)
      -- normalize y locally:
      yuv[{1}] = normalization(yuv[{{1}}])
      testData.data[i] = yuv
   end
   -- normalize u globally:
   testData.data[{ {},2,{},{} }]:add(-mean_u)
   testData.data[{ {},2,{},{} }]:div(-std_u)
   -- normalize v globally:
   testData.data[{ {},3,{},{} }]:add(-mean_v)
   testData.data[{ {},3,{},{} }]:div(-std_v)
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> verify statistics')

-- It's always good practice to verify that data is properly
-- normalized.

--for i,channel in ipairs(channels) do
--   local trainMean = trainData.data[{ {},i }]:mean()
--   local trainStd = trainData.data[{ {},i }]:std()
--
--   local testMean = testData.data[{ {},i }]:mean()
--   local testStd = testData.data[{ {},i }]:std()
--
--   print('training data, '..channel..'-channel, mean: ' .. trainMean)
--   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)
--
--   print('test data, '..channel..'-channel, mean: ' .. testMean)
--   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
--end
--
----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> visualizing data')

-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

--if opt.visualize then
--   local first256Samples_y = trainData.data[{ {1,256},1 }]
--   image.display{image=first256Samples_y, nrow=16, legend='Some training examples: Y channel'}
--   local first256Samples_y = testData.data[{ {1,256},1 }]
--   image.display{image=first256Samples_y, nrow=16, legend='Some testing examples: Y channel'}
--end

-- Exports
result = {
   trainData = trainData,
   testData = testData,
   mean = mean,
   std = std,
   classes = classes
}
torch.save("traindata.data",result)
return result

