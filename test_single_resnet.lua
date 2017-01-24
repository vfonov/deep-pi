require 'nn'
require 'image'
local imagenetLabel = require './imagenet_resnet'

local topk=5
local test_image_file=arg[1]


torch.setdefaulttensortype('torch.FloatTensor')
-- load pre-trained model
local prefix=os.getenv("HOME")..'/'
local t = torch.Timer()
local m=torch.load(prefix..'resnet-18-cpu_arm.t7')
print(string.format("loading model: %.2fsec",t:time().real))

local model=m
--print(model)
--local mean_std=model.transform
local mean_std = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

-- add soft max layer, to ouput pseudo-probabilities
model:add(nn.LogSoftMax())
model:evaluate()
--local words=torch.load(prefix..'words_1000_ascii.t7','ascii')

-- load test image
test_image_file=test_image_file or prefix.."n07579787_ILSVRC2012_val_00049211.JPEG"

print("Using test image: "..test_image_file)
local input=image.load(test_image_file)

-- model input size
local oW=224
local oH=224

-- find the smaller dimension, and resize it to loadSize (while keeping aspect ratio)
if input:size(3) < input:size(2) then
   input = image.scale(input, oW, oH * input:size(2) / input:size(3))
else
   input = image.scale(input, oW * input:size(3) / input:size(2), oH)
end

local w1 = math.ceil((input:size(3)-oW)/2)
local h1 = math.ceil((input:size(2)-oH)/2)

local cropped = image.crop(input, w1, h1, w1+oW, h1+oH) -- center patch

-- perform normalization (remove pre-trained mean and std)
for i=1,3 do
  cropped[{{i},{},{}}]:add(-mean_std.mean[i])
  cropped[{{i},{},{}}]:div(mean_std.std[i])
end

-- add a fake dimension of size 1
cropped=cropped:view(1,3,oH,oW)

t = torch.Timer() 
local output=model:forward(cropped)
print(string.format("Running neural net: %.2fsec",t:time().real))

-- extract topK classes:
output=output:view(1000)
local output_x, output_sorted = output:sort(1,true)
probs=torch.exp(output_x)

for i=1,topk do
  print(string.format(" %0.1f%%: %s",probs[i]*100,imagenetLabel[ output_sorted[i] ]) )
end

