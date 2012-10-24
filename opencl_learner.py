"""
opencl_learner
===

Implements OpenCLNetwork providing OpenCL
based functions to calculate neural network passes
on dataset batches. 

@contrastive_divergence currently disabled

"""


import pyopencl as cl
import numpy
import pyopencl.clrandom
import pyopencl.array
from mako.template import Template
import math


class OpenCLNetwork:
	def __init__(self, opencl_context, network, batch_size, use_double=False):
		self.ctx = opencl_context
		self.network = network

		self.queue = cl.CommandQueue(self.ctx)
		self.batch_size = batch_size
		self.program = None

		self.memory_flags = cl.mem_flags.READ_WRITE# | cl.mem_flags.ALLOC_HOST_PTR

		self.use_double = use_double
		if False:
			self.float_size = 8
			self.float_type = numpy.float64
			self.float_name = 'double'
		else:
			self.float_size = 4
			self.float_type = numpy.float32
			self.float_name = 'float'

		self.states = cl.Buffer(self.ctx, self.memory_flags, self.float_size*self.network.layers_sum*batch_size )
		self.batches_buf = cl.Buffer(self.ctx, self.memory_flags, self.float_size*(self.network.sizes[0]+self.network.sizes[-1])*batch_size )
		self.R_states = cl.Buffer(self.ctx, self.memory_flags, self.float_size*self.network.layers_sum*batch_size )

		self.LJ = cl.Buffer(self.ctx, self.memory_flags, self.float_size*self.network.sizes[-1]*batch_size )
		self.error = cl.Buffer(self.ctx, self.memory_flags, self.float_size*self.network.sizes[-1]*batch_size )
		self.targets = cl.Buffer(self.ctx, self.memory_flags, self.float_size*self.network.sizes[-1]*batch_size )

		self.temp_vec1 = cl.Buffer(self.ctx, self.memory_flags, self.float_size*self.network.layer_max*batch_size*2 )
		self.temp_vec2 = (self.temp_vec1,cl.Buffer(self.ctx, self.memory_flags, self.float_size*self.network.layer_max*batch_size*2 ))

		self.loss_summer = cl.Buffer(self.ctx, self.memory_flags, self.float_size*batch_size )

		self.weights = cl.Buffer(self.ctx, self.memory_flags, self.float_size*self.network.weights_sum) 
		self.R_weights = cl.Buffer(self.ctx, self.memory_flags, self.float_size*self.network.weights_sum) 
		self.result_weights = cl.Buffer(self.ctx, self.memory_flags, self.float_size*self.network.weights_sum) 


		self.current_batch = None

		self.work_group_size = 16

		self.load_programs()
		self.update_weights(network)

	def update_weights(self,network=None,gpu_net_array=None):
		if gpu_net_array is None:
			gpu_net_array = network.to_array(order='F')
		cl.enqueue_copy(self.queue, self.weights, gpu_net_array,is_blocking=False)
	def update_R_weights(self,network=None,gpu_net_array=None):
		if gpu_net_array is None:
			gpu_net_array = network.to_array(order='F')
		cl.enqueue_copy(self.queue, self.R_weights, gpu_net_array,is_blocking=False)

	def read_result_weights(self, gpu_net_array):
		cl.enqueue_copy(self.queue, gpu_net_array, self.result_weights, is_blocking=True)
		return gpu_net_array

	def read_weights(self, gpu_net_array):
		cl.enqueue_copy(self.queue, gpu_net_array, self.weights, is_blocking=True)
		return gpu_net_array

	
	def load_programs(self):
		f = open("program.cl",'r')
		program_tmp = f.read()
		program_src = str(Template(program_tmp).render(
			layer_count=len(self.network.sizes),
			weight_count=len(self.network.sizes)-1,
			layer_sizes=self.network.sizes,
			weight_sizes=self.network.weights_sizes,
			layer_offsets=self.network.layer_offsets,
			weight_offsets=self.network.weights_offsets,
			work_group_size=self.work_group_size,
			vector_count = self.batch_size,
			float_type = ['float','double'][self.use_double],
			use_double = self.use_double
			))

		self.program = cl.Program(self.ctx, program_src).build("-Werror")

	def load_batch(self, batch):
		cl.enqueue_copy(self.queue, self.batches_buf, batch.buffers,is_blocking=False)
		self.program.copy_buffers(self.queue,(self.work_group_size*self.network.sizes[0],),(self.work_group_size,),
			self.batches_buf,self.states,self.targets)

	def forward_pass(self):
		for x in xrange(len(self.network.weights_sizes)):
			self.program.forward_pass(self.queue,(self.work_group_size*self.network.layer_max,),(self.work_group_size,),
				self.states,
				self.weights,
				numpy.uint32(x))

	def R_forward_pass(self):
		for x in xrange(len(self.network.weights_sizes)):	
			self.program.R_forward_pass(self.queue,(self.work_group_size*self.network.layer_max,),(self.work_group_size,),
				self.states,
				self.R_states,
				self.weights,
				self.R_weights,
				self.LJ,
				numpy.uint32(x))
	
	def backward_pass(self,vector):
		self.program.zero(self.queue,(self.network.weights_sum,),None,
			self.result_weights
		)
		self.program.backward_pass(self.queue,(self.work_group_size*self.network.layer_max,),(self.work_group_size,),
			self.states,
			vector,
			self.weights,
			self.result_weights,
			self.temp_vec2[(len(self.network.weights_sizes)) % 2],
			numpy.uint32(len(self.network.weights_sizes)-1)
			)
		for x in reversed(xrange(len(self.network.weights_sizes)-1)):
			self.program.backward_pass(self.queue,(self.work_group_size*self.network.layer_max,),(self.work_group_size,),
				self.states,
				self.temp_vec2[(x)%2],
				self.weights,
				self.result_weights,
				self.temp_vec2[(x+1)%2],
				numpy.uint32(x)
				)

	def gradient(self, batch):
		self.load_batch(batch)
		self.forward_pass()
		self.program.load_error(self.queue,(self.work_group_size*self.network.layer_max,),(self.work_group_size,),
			self.states.get_sub_region(self.float_size*self.batch_size*self.network.layer_offsets[-2],self.float_size*self.batch_size*self.network.sizes[-1]),
			self.targets,
			self.error
		)
		self.backward_pass(self.error)
		self.program.calc_loss(self.queue,(self.work_group_size*self.network.layer_max,),(self.work_group_size,),
			self.states,
			self.weights,
			self.targets,
			self.loss_summer)

	def read_back_loss(self):
		self.program.sum(self.queue,(self.work_group_size*1,),(self.work_group_size,),
			self.loss_summer,
			numpy.uint32(self.batch_size),
			numpy.uint32(self.batch_size))
		loss_result = numpy.array([0.5]).astype(self.float_type)
		cl.enqueue_copy(self.queue, loss_result, 
			self.loss_summer,
			is_blocking=True)
		return loss_result[0]

	def load_forward_and_loss(self, batch):
		self.load_batch(batch)
		self.forward_pass()
		self.program.calc_loss(self.queue,(self.work_group_size*self.network.layer_max,),(self.work_group_size,),
			self.states,
			self.weights,
			self.targets,
			self.loss_summer)


	def gauss_product(self):
		self.R_forward_pass()
		self.backward_pass(self.LJ)

	# def contrastive_divergence(self, batch, rbm):
	# 	self.load_batch(batch)
	# 	self.forward_pass()
	# 	rnd_size = 2**int(math.ceil(math.log(self.network.weights_sum, 2)))

	# 	vec = pyopencl.clrandom.rand(self.queue, (rnd_size,), dtype=numpy.float32)

	# 	self.program.contrastive_divergence(self.queue,(self.work_group_size*self.network.layer_max,),(self.work_group_size,),
	# 		self.states,
	# 		self.temp_vec1,
	# 		self.weights,
	# 		self.result_weights,
	# 		vec.data,
	# 		numpy.uint32(rnd_size), 
	# 		numpy.uint32(rbm), 
	# 		numpy.uint32(3), 
	# 		numpy.uint32(0.1)
	# 		)

	def zero_results(self):
		self.program.zero(self.queue,(self.network.weights_sum,),None,
			self.result_weights
		)