from torchvision import transforms
from .encoding import Intensity2Latency
from utils import functional as sf

class InputTransform:
    def __init__(self, filter, timesteps = 15):
        self.to_tensor = transforms.ToTensor()
        self.filter = filter
        if self.filter.use_abs == True:
            self.temporal_transform = Intensity2Latency(timesteps, to_spike=True)
        else:
            self.temporal_transform = Intensity2Latency(timesteps)
        self.cnt = 0
        
    def __call__(self, image):
        image = self.to_tensor(image) * 255
        image.unsqueeze_(0)
        image = self.filter(image)
        if self.filter.use_abs == True:
            image = sf.pointwise_inhibition(image)
        else:
            image = sf.local_normalization(image, 8)
        temporal_image = self.temporal_transform(image)
        return temporal_image


class S1C1Transform:
	def __init__(self, filter, timesteps = 15):
		self.to_tensor = transforms.ToTensor()
		self.filter = filter
		self.temporal_transform = Intensity2Latency(timesteps)
		self.cnt = 0

	def __call__(self, image):
		image = self.to_tensor(image) * 255
		image.unsqueeze_(0)
		image = self.filter(image)
		image = sf.local_normalization(image, 8)
		temporal_image = self.temporal_transform(image)
		return temporal_image.sign().byte()