require 'image'
require 'camera' -- for camera
display = require 'display' -- for displaying 

local display_sample_in={}
cam = image.Camera {idx=0,width=640,height=480}

while true do
    frame = cam:forward()
    display_sample_in.win=display.image(frame,display_sample_in)
end
