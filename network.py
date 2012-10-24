"""
network.py
===

The DeepBeliefNetwork class provides a CPU implementation
of the neural network functions. The methods are unbatched
in comparison to the GPU implementation. This class also provides
a method of initialization, that can be used in conjunction with the
gpu implementation.

"""

import numpy
import pyopencl as cl

def sigmoid(x):
	return numpy.reciprocal( 1.0 + numpy.nan_to_num(numpy.exp(-x)) )

def sigmoid_gradient_y(y):
	return y * (1.0 - y)
def sigmoid_gradient(input, target):
	return (input- target)

def sigmoid_H_product(X, P):
	return X * P * (1-P)

def sigmoid_loss(output, target):
	v = -(target*numpy.log(output) + (numpy.ones_like(target)-target)*numpy.nan_to_num(numpy.log1p(-output)))

	return numpy.nan_to_num(v).sum()


class DeepBeliefNetwork:
	def __init__(self, sizes, initialize=True):
		self.sizes 	= sizes
	
		self.weights_sizes = [(v+1)*h for (v, h) in zip(self.sizes[:-1],self.sizes[1:])]
		self.weights_sum = sum((v+1)*h for (v, h) in zip(self.sizes[:-1],self.sizes[1:]))
		self.layers_sum = sum(self.sizes)
		self.layer_max = max(self.sizes)
		self.weights_offsets = [0]+list(self.weights_sizes)
		self.layer_offsets = [0]+list(self.sizes)

		for i in range(len(self.weights_offsets)-1):
			i=i+1
			self.weights_offsets[i] += self.weights_offsets[i-1]

		for i in range(len(self.layer_offsets)-1):
			i=i+1
			self.layer_offsets[i] += self.layer_offsets[i-1]

		self.w = [ numpy.zeros((v+1,h)) #.astype(numpy.float32)
								for (v, h) in zip(sizes[:-1],
												  sizes[1:])]
		if initialize:
			for i,w in enumerate(self.w):

				stepping = min(1, w.shape[1] // w[1].shape[0])
				width = stepping*4
				rt_width = float(width)**0.5

				for hid in range(w.shape[1]):
					for vis in range(stepping*hid, stepping*hid + width):
						if vis >= w.shape[0]-1:#Ignore bias
							break
						self.w[i][vis][hid] = numpy.random.normal(0.0,1./rt_width)
									
	
	def save(self,filename):
		import gzip
		f = gzip.open(filename,'w',9)
		items = numpy.concatenate([x.flatten().tolist() for x in self.w]).astype(numpy.float64)
		f.write(items.tostring())
		f.close()

	def load(self,filename):
		import gzip
		f = gzip.open(filename,'r')
		str = f.read()
		items = numpy.fromstring(str, dtype=numpy.float64)

		offsets = [0]+[(v+1)*h for (v, h) in zip(self.sizes[:-1],self.sizes[1:])]

		for i in range(len(offsets)-1):
			i=i+1
			offsets[i] += offsets[i-1]

		self.w = [ numpy.array(items[offsets[i]:offsets[i+1]]).reshape((v+1,h))#.astype(numpy.float64)
					for i,(v, h) in enumerate(zip(self.sizes[:-1],
										self.sizes[1:]))]

	def forward_pass(self, V):
		state = [None] * len(self.sizes)

		state[0] = numpy.array(V)
		for i in range(len(self.sizes) - 1):
			X = numpy.append(state[i],1.0)
			state[i + 1] = sigmoid( numpy.dot(X, self.w[i]) )

		return state

	def predictions(self, input):
		state = self.forward_pass(input)
		return state[-1]

	def R_forward_pass(self, state, v):
		R_state_X = [None] * len(self.sizes)

		R_state_X[0] = numpy.array(state[0]*0)

		R_state_i = R_state_X[0]
		for i in range(len(self.sizes) - 1):
			R_state_X[i+1] =  numpy.dot(numpy.append(state[i],1.0), v.w[i]) + \
							  numpy.dot(numpy.append(R_state_i,0.0), self.w[i])

			R_state_i = sigmoid_gradient_y(state[i+1]) * R_state_X[i+1]

		return R_state_X [-1]

	def backward_pass(self, state, doutput):
		grad = DeepBeliefNetwork( self.sizes, initialize=False )

		dY = doutput
		for i in reversed(range(len(self.sizes) - 1)):
			dX = sigmoid_gradient_y(state[i + 1]) * dY
			X = state[i]
			grad.w[i] += numpy.outer(numpy.append(X,1.0), dX)

			## backprop the gradient:
			if i > 0: # typically the first multiplication is the costliest.
				dY = numpy.dot(dX, self.w[i].T)[:-1]

		return grad

	def array_like(self, order='F', dtype=numpy.float32):
		return numpy.zeros( sum( x.size for x in self.w ) ).astype(dtype)

	def to_array(self, order='F', dtype=numpy.float32): # order F for gpu
		return numpy.concatenate([numpy.array(x).flatten(order=order) for x in self.w]).astype(dtype)

	def from_array(self, data, order='F', dtype=numpy.float32): # order F for gpu
		offsets = [0]+[(v+1)*h for (v, h) in zip(self.sizes[:-1],self.sizes[1:])]

		for i in range(len(offsets)-1):
			i=i+1
			offsets[i] += offsets[i-1]

		self.w = [ numpy.array(data[offsets[i]:offsets[i+1]]).reshape((v+1,h),order=order).astype(dtype)
					for i,(v, h) in enumerate(zip(self.sizes[:-1],
										self.sizes[1:]))]

		return self

	def gradient_loss(self, input, target):
		state = self.forward_pass(input)
		output = state[-1]

		return sigmoid_gradient(output, target)

	def gradient(self, input, target):
		state = self.forward_pass(input)
		output = state[-1]

		loss = sigmoid_loss(output, target)
		doutput = sigmoid_gradient(output, target)
		probability = output#self.af[-1].apply(output)

		grad = self.backward_pass(state, doutput) 
		return grad.to_array('F')#grad.pack()#, loss, state#output

	def gauss_newton(self, input, target, v, state=None):
		if state is None:
			state = self.forward_pass(input)

		R_output = self.R_forward_pass(state, v)
		output = state[-1]

		predictions = output#self.af[-1].apply(output)
		LJ = sigmoid_H_product(R_output, predictions)

		return self.backward_pass(state, LJ).to_array('F')

	def loss(self, input, target):
		state = self.forward_pass(input)
		output = state[-1]

		return sigmoid_loss(output, target)