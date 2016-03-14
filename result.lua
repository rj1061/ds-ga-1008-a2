require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'cunn'

dofile './csv.lua'
----------------------------------------------------------------------
print '==> loading model'
modelpath = './logs/pseudolabels/model.net'


model = torch.load(modelpath)
model:evaluate()


----------------------------------------------------------------------
print '==> loading dataset'

dofile 'provider.lua'
provider = torch.load('provider.t7')

print('==> testing on test set:')

-- classes
classes = {'1','2','3','4','5','6','7','8','9','0'}

-- making prediction and write to csv
separator = ','  -- optional; use if not a comma
csv = Csv("predictions.csv", "w", separator)
csv:write({"Id","Prediction"}) -- write header


bs = 25
for i = 1,provider.testData:size(), bs do
        -- disp progress
        xlua.progress(i, provider.testData:size())

        -- get new sample
	local outputs = model:forward(provider.testData.data:narrow(1,i,bs):cuda())

        -- test sample
        local pred, idx = torch.max(outputs,2)
	for j =1,bs do
       	 csv:write({i+j-1,idx[j][1]}) -- write each data row
	end 
end

csv:close()
