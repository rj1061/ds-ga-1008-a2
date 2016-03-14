require 'nn'
require 'image'
require 'xlua'

torch.setdefaulttensortype('torch.FloatTensor')

-- parse STL-10 data from table into Tensor
function parseDataLabel(d, numSamples, numChannels, height, width)
   local t = torch.ByteTensor(numSamples, numChannels, height, width)
   local l = torch.ByteTensor(numSamples)
   local idx = 1
   for i = 1, #d do
      local this_d = d[i]
      for j = 1, #this_d do
    t[idx]:copy(this_d[j])
    l[idx] = i
    idx = idx + 1
      end
   end
   assert(idx == numSamples+1)
   return t, l
end

function parseDataNoLabel(d, numSamples, numChannels, height, width)
  local t_nl = torch.ByteTensor(numSamples, numChannels, height, width)
  local idx = 1
  for i = 1, #d do
     local this_d = d[i]
     for j = 1, #this_d do
   t_nl[idx]:copy(this_d[j])
   idx = idx + 1
     end
  end
  assert(idx == numSamples + 1)
  return t_nl
end


local Provider = torch.class 'Provider'

function Provider:__init(full)
  local trsize = 4000
  local valsize = 1000  -- Use the validation here as the valing set
  local testsize = 8000
  local channel = 3
  local height = 96
  local width = 96

  -- download dataset
  if not paths.dirp('stl-10') then
     os.execute('mkdir stl-10')
     local www = {
         train = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/train.t7b',
         val = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/val.t7b',
         extra = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/extra.t7b',
         test = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/test.t7b'
     }

     os.execute('wget ' .. www.train .. '; '.. 'mv train.t7b stl-10/train.t7b')
     os.execute('wget ' .. www.val .. '; '.. 'mv val.t7b stl-10/val.t7b')
     os.execute('wget ' .. www.test .. '; '.. 'mv test.t7b stl-10/test.t7b')
     os.execute('wget ' .. www.extra .. '; '.. 'mv extra.t7b stl-10/extra.t7b')
  end

  local raw_train = torch.load('stl-10/train.t7b')
  local raw_val = torch.load('stl-10/val.t7b')
  local raw_test = torch.load('stl-10/test.t7b')

  -- load and parse dataset
  self.testData = {
     data = torch.Tensor(),
     labels = torch.Tensor(),
     size = function() return testsize end
  }
  self.testData.data, self.testData.labels = parseDataLabel(raw_test.data, testsize, channel, height, width)
  local testData = self.testData

  self.trainData = {
     data = torch.Tensor(),
     labels = torch.Tensor(),
     size = function() return trsize end
  }
  self.trainData.data, self.trainData.labels = parseDataLabel(raw_train.data,
                                                   trsize, channel, height, width)
  local trainData = self.trainData
  self.valData = {
     data = torch.Tensor(),
     labels = torch.Tensor(),
     size = function() return valsize end
  }
  self.valData.data, self.valData.labels = parseDataLabel(raw_val.data,
                                                 valsize, channel, height, width)
  local valData = self.valData

  -- convert from ByteTensor to Float
  self.testData.data = self.testData.data:float()
  self.testData.labels = self.testData.labels:float()
  self.trainData.data = self.trainData.data:float()
  self.trainData.labels = self.trainData.labels:float()
  self.valData.data = self.valData.data:float()
  self.valData.labels = self.valData.labels:float()
  collectgarbage()
end

function Provider:getUnlabelled()
  local exsize = 100000
  if not paths.dirp('stl-10') then
     local www = {
         extra = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/extra.t7b',
     }

     os.execute('wget ' .. www.extra .. '; '.. 'mv extra.t7b stl-10/extra.t7b')
  end
  local raw_extra = torch.load('stl-10/extra.t7b')
  local extraData = {
    data = torch.FloatTensor(),
    size = function() return exsize end
  }

  extraData.data = parseDataNoLabel(raw_extra.data, 100000, 3, 96, 96)
  -- local extraData = self.extraData
  extraData.data = extraData.data:float()

  self.extraData = extraData
  collectgarbage()
end

function Provider:normalize()
  ----------------------------------------------------------------------
  -- preprocess/normalize train/val sets
  --
  local trainData = self.trainData
  local valData = self.valData
  local extraData = self.extraData
  local testData = self.testData

  print '<trainer> preprocessing data (color space + normalization)'
  collectgarbage()

  -- preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,trainData:size() do
     xlua.progress(i, trainData:size())
     -- rgb -> yuv
     local rgb = trainData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     trainData.data[i] = yuv
  end
  -- normalize u globally:
  local mean_u = trainData.data:select(2,2):mean()
  local std_u = trainData.data:select(2,2):std()
  trainData.data:select(2,2):add(-mean_u)
  trainData.data:select(2,2):div(std_u)
  -- normalize v globally:
  local mean_v = trainData.data:select(2,3):mean()
  local std_v = trainData.data:select(2,3):std()
  trainData.data:select(2,3):add(-mean_v)
  trainData.data:select(2,3):div(std_v)

  trainData.mean_u = mean_u
  trainData.std_u = std_u
  trainData.mean_v = mean_v
  trainData.std_v = std_v

  -- preprocess valSet
  for i = 1,valData:size() do
    xlua.progress(i, valData:size())
     -- rgb -> yuv
     local rgb = valData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     valData.data[i] = yuv
  end
  -- normalize u globally:
  valData.data:select(2,2):add(-mean_u)
  valData.data:select(2,2):div(std_u)
  -- normalize v globally:
  valData.data:select(2,3):add(-mean_v)
  valData.data:select(2,3):div(std_v)

  -- preprocess testSet
  for i = 1,testData:size() do
    xlua.progress(i, testData:size())
     -- rgb -> yuv
     local rgb = testData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     testData.data[i] = yuv
  end
  -- normalize u globally:
  testData.data:select(2,2):add(-mean_u)
  testData.data:select(2,2):div(std_u)
  -- normalize v globally:
  testData.data:select(2,3):add(-mean_v)
  testData.data:select(2,3):div(std_v)

  -- preprocess extraSet
  for i=1, extraData:size() do
    xlua.progress(i, extraData:size())
    -- rgb -> yuv
    local rgb = extraData.data[i]
    local yuv = image.rgb2yuv(rgb)
    -- normalize y locally:
    yuv[{1}] = normalization(yuv[{{1}}])
    extraData.data[i] = yuv
  end
  extraData.data:select(2,2):add(-mean_u)
  extraData.data:select(2,2):div(std_u)

  extraData.data:select(2,3):add(-mean_v)
  extraData.data:select(2,3):div(std_v)
end
