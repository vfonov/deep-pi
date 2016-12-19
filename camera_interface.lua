-- useOpenCV=true
require 'nn'
require 'image'
require 'camera' -- for camera
display = require 'display' -- for displaying 


local display_sample_in={}
local display_output={}

torch.setdefaulttensortype('torch.FloatTensor')

-- load pre-trained model
local prefix="/home/vfonov/"

local t = torch.Timer()
local m=torch.load(prefix..'nin_bn_final_arm.t7')
print(string.format("loading model:%.2fsec",t:time().real))

local model=m:unpack()

-- add soft max layer, to ouput pseudo-probabilities
model:add(nn.LogSoftMax())
-- switch model to evaluate mode
model:evaluate()

local classes=model.classes
local words=torch.load(prefix..'words_1000_ascii.t7','ascii')

local mean_std=model.transform

-- input (from camera)
local iW=320
local iH=240

-- output (for model)
local oW=224
local oH=224

local topk=5

-- parameters for cropping:
local w1 = math.ceil((iW-oW)/2)
local h1 = math.ceil((iH-oH)/2)

local cam = image.Camera {idx=0,width=iW,height=iH}

-- starting infinte loop
while true do

  local frame = cam:forward()
  local cropped = image.crop(frame, w1, h1, w1+oW, h1+oH) -- center patch

  -- perform normalization (remove pre-trained mean and std)
  for i=1,3 do
    cropped[{{i},{},{}}]:add(-mean_std.mean[i])
    cropped[{{i},{},{}}]:div(mean_std.std[i])
  end

  display_sample_in.win=display.image(cropped,display_sample_in)

  -- add a fake dimension of size 1
  cropped=cropped:view(1,3,oH,oW)
  local output=model:forward(cropped)

  -- extract topK classes:
  output=output:view(1000)
  local output_x, output_sorted = output:sort(1,true)
  probs=torch.exp(output_x)

  local out_text=""
  for i=1,topk do
   out_text=out_text..string.format(" %0.1f:%s <br />\n ",probs[i]*100,words[ classes[ output_sorted[i] ] ])
  end

  display_output.win=display.text(out_text,display_output)
  -- print output in the terminal too
  print(out_text)

end
