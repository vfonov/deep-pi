require 'nn'
require 'image'


local topk=5
local test_image_file=arg[1]


torch.setdefaulttensortype('torch.FloatTensor')
-- load pre-trained model
local prefix=os.getenv("HOME")..'/'
local t = torch.Timer()
local m=torch.load(prefix..'nin_bn_final_arm.t7')
print(string.format("loading model: %.2fsec",t:time().real))

local model=m:unpack()
local classes=model.classes
local mean_std=model.transform

-- replace the first layer to work with grayscale images
local old_first=model:get(1)
local new_first=nn.SpatialConvolution(1, 96, 11, 11, 4,4, 5,5)

new_first.weight[{{},{},{},{}}]=old_first.weight[{{},1,{},{}}]+
                                old_first.weight[{{},2,{},{}}]+
                                old_first.weight[{{},3,{},{}}]

new_first.gradWeight[{{},{},{},{}}]=old_first.gradWeight[{{},1,{},{}}]+
                                    old_first.gradWeight[{{},2,{},{}}]+
                                    old_first.gradWeight[{{},3,{},{}}]
                                
model:remove(1)
model:insert(new_first,1)

-- add soft max layer, to ouput pseudo-probabilities
model:add(nn.LogSoftMax())
model:evaluate()
local words=torch.load(prefix..'words_1000_ascii.t7','ascii')

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
cropped_bw=torch.Tensor(1,1,oH,oW)

cropped_bw[ {{},{},{},{}} ]=(cropped[ {{},1,{},{}} ]+
                             cropped[ {{},2,{},{}} ]+
                             cropped[ {{},3,{},{}} ] )/3
                             
print(torch.type(cropped_bw))

image.save("test.jpg",(cropped_bw:view(1,oH,oW)+1.0)/2.0)

t = torch.Timer() 
local output=model:forward(cropped_bw)
print(string.format("Running neural net: %.2fsec",t:time().real))

-- extract topK classes:
output=output:view(1000)
local output_x, output_sorted = output:sort(1,true)
probs=torch.exp(output_x)

for i=1,topk do
  print(string.format(" %0.1f%%: %s: %s ",probs[i]*100,classes[ output_sorted[i] ], words[ classes[ output_sorted[i] ] ]) )
end

