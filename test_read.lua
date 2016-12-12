require 'paths'
require 'sys'
require 'minc2_simple'

dataset={}

for i=0,1 do
for j=1,1000 do

        local t1=minc2_file.new(string.format('temp/%02d/scan_%03d.mnc',i,j))
        t1:setup_standard_order()
        
        local seg=minc2_file.new(string.format('temp/%02d/seg_%03d.mnc',i,j))
        seg:setup_standard_order()
        
        dataset[#dataset+1]={ t1:load_complete_volume(minc2_file.MINC2_FLOAT), 
                             seg:load_complete_volume(minc2_file.MINC2_INT) }
        -- print(collectgarbage("count"))
        --collectgarbage()
        t1:close()
        seg:close()        
end
end


