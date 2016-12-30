require 'nn'
require 'image'

local n_iter = 10
local n_end_layer = 5  -- max 43 for NiN
local step_size=0.1
local clip=true

torch.setdefaulttensortype('torch.FloatTensor')

local input_img='frame.jpg' -- 
-- load pre-trained model
local prefix=os.getenv("HOME")..'/'

local m=torch.load( prefix..'nin_bn_final_arm.t7' )
net=m:unpack()
cls=m.classes


local Normalization=net.transform

function reduceNet(full_net, end_layer)
    local net = nn.Sequential()
    if end_layer>=#full_net then
      return full_net
    end
    for l=1,end_layer do
        net:add(full_net:get(l))
    end
    --net:evaluate()
    net:training()
    return net
end


function make_step(net, img, clip, step_size)
    local step_size = step_size or 0.01
    local clip = clip
    if clip == nil then clip = true end

    local dst, g
    
    local cpu_img = img:view(1,img:size(1),img:size(2),img:size(3))
    dst = net:forward(cpu_img)
    g   = net:updateGradInput(cpu_img,dst):squeeze()
    
    -- apply normalized ascent step to the input image
    img:add(g:mul(step_size/torch.abs(g):mean()))
    
    if clip then
      local i
      for i=1,3 do
        local bias = Normalization.mean[i]/Normalization.std[i]
        img[{i,{},{}}]:clamp(-bias,1/Normalization.std[i]-bias)
      end
    end
    return img
end

function deepdream(net, base_img, iter_n, clip, step_size)
    local iter_n = iter_n
    local net = net
    local step_size = step_size
    
    local clip = clip
    -- prepare base images for all octaves
    local i
    
    local img=base_img:clone()
    
    for i=1,3 do
     img[{{i},{},{}}]:add(-Normalization.mean[i])
     img[{{i},{},{}}]:div(Normalization.std[i])
    end

    src = img
    
    for i=1,iter_n do
        src = make_step(net, src, clip, step_size)
    end
    
    -- returning the resulting image
    for i=1,3 do
      src[{i,{},{}}]:mul(Normalization.std[i]):add(Normalization.mean[i])
    end
    
    return src
end

local t = torch.Timer()
if n_end_layer then
    net = reduceNet(net, n_end_layer)
end
print(string.format("reducing net: %.2fsec",t:time().real))

print(net)
--
img = image.load(input_img)
--
local t = torch.Timer()
x = deepdream(net, img, n_iter, clip, step_size)
print(string.format("dreaming: %.2fsec",t:time().real))
--image.display(x)
image.save('test.jpg',x)
