-- useOpenCV=true
require 'image'
require 'camera' -- for camera
display = require 'display' -- for displaying 

local display_sample_in={}
cam = image.Camera {idx=0,width=320,height=240}

while true do
    frame = cam:forward()
    display_sample_in.win=display.image(frame,display_sample_in)
end
