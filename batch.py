"""
Some batching functions.
"""


import numpy
import numpy.random

class Batch:
	def __init__(self, arrays, input_size, output_size):
		self.size = len(arrays)
		self.buffers = arrays
		self.input_size = input_size
		self.output_size = output_size

	def sub_batch(self,from_, to):
		return Batch(self.buffers[from_:to],self.input_size,self.output_size)

	def split(self, parts):
		part_size = self.size // parts
		return[ self.sub_batch(xz*part_size, (xz+1)*part_size) for xz in xrange(parts) ]

	def sample(self, count):
		count = min(count, self.size)
		randomized = numpy.random.permutation(self.size)
		return Batch(numpy.take(self.buffers,randomized[:count],axis=0),self.input_size,self.output_size)

	def sample_split(self, samples, parts):
		return self.sample(samples).split(parts)
