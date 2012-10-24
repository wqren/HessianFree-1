/*
program.cl
===

OpenCL functions for network passes.
All functions are templated for some unrolling
and constant expansion.
If 64 bit floating points are available, they
will be used for accumulation and the sigmoid
function.

Function list:
---

@sigmoid: calculates the sigmoid function
@forward_pass: forward pass through the network
@R_forward_pass: R op pass through the network
@backward_pass: ...
@copy_buffers:
@zero:

@rbm_step_forward
@rbm_step_back

@calc_loss
@sum
@load_error
@gradient_y
@H_product
@gradient
@layer_size
@weight_size
@layer_offset
@weight_offset

@contrastive_divergence: Currently has race conditions on global memory, and so is disabled.
*/

% if use_double:
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
% endif

// 32 bit floats are used for storage,
// but if 64 bit floats are available we use them for
// accumulation and sigmoid
typedef float float_t; 

% if use_double:
inline float_t sigmoid(double x)
{
	return (float_t)(1.0 / ( 1.0 + exp(-x )));
% endif:
% if not use_double:
inline float_t sigmoid(float_t x)
{
	return 1.0 / ( 1.0 + exp(-x) );
% endif:
}

inline float_t gradient_y(float_t y)
{
	return y * (1.0 - y);
}

inline float_t H_product(float_t x, float_t p)
{
	return x * p * (1.0-p);
}

inline float_t gradient(float_t input, float_t target)
{
	return (input - target);
}

inline size_t layer_size(size_t i)
{
% for i, size in enumerate(layer_sizes):
	if( ${i} == i) return ${size};
% endfor
	return 0;
}

inline size_t weight_size(size_t i)
{
% for i, size in enumerate(weight_sizes):
	if( ${i} == i) return ${size};
% endfor
	return 0;
}

inline size_t layer_offset(size_t i)
{
% for i, size in enumerate(layer_offsets):
	if( ${i} == i) return ${size};
% endfor
	return 0;
}

inline size_t weight_offset(size_t i)
{
% for i, size in enumerate(weight_offsets):
	if( ${i} == i) return ${size};
% endfor
	return 0;
}


__attribute__((reqd_work_group_size( ${work_group_size} , 1, 1)))
__kernel void forward_pass(	
	__global float_t *state_arrays,

	__global const float_t *weight_matrix,

	const uint layer
	)
{
	__local ${float_type} local_store[ ${work_group_size} ];

	const uint gidx = get_global_id(0);
	const uint lidx = get_local_id (0);

	const uint vector_count = ${ vector_count };

	{
		uint vector_id = 0;
		const size_t visible_size = layer_size(layer);
		const size_t hidden_size = layer_size(layer+1);
		const size_t visible_offset = layer_offset(layer)*vector_count;
		const size_t hidden_offset = layer_offset(layer+1)*vector_count;
		const size_t weights_offsets = weight_offset(layer);

		size_t hidden_id = get_group_id(0);
		for (uint id = hidden_id; id < hidden_size * vector_count; id += get_num_groups(0))
		{
			hidden_id=id%hidden_size;
			vector_id=id/hidden_size;
			float_t val=0.0;

			uint x = lidx;
			for (; x < visible_size; x += get_local_size(0))
			{
				val += state_arrays[visible_offset + x + visible_size*vector_id] * weight_matrix[weights_offsets + x + (visible_size+1)*hidden_id];
			}

			if(x == visible_size)
				val += weight_matrix[weights_offsets + visible_size + (visible_size+1)*hidden_id];

			local_store[lidx]=val;

			for (uint stride = get_local_size(0)/2; stride > 0; stride /= 2)
			{
				barrier(CLK_LOCAL_MEM_FENCE);
				if (lidx < stride)
					local_store[lidx] += local_store[lidx + stride];
			}

			if (lidx == 0)
			{		
				state_arrays[hidden_offset + hidden_id + vector_id*hidden_size] = sigmoid(local_store[lidx]);
			}
		}
		// barrier(CLK_GLOBAL_MEM_FENCE);
	}
}


__attribute__((reqd_work_group_size( ${work_group_size} , 1, 1)))
__kernel void R_forward_pass(	
	__global float_t *state_arrays,
	__global float_t *R_state_arrays,

	__global const float_t *weight_matrix,
	__global const float_t *R_weight_matrix,

	__global float_t *LJ,

	const uint layer
	)
{
	__local ${float_type} local_store[ ${work_group_size} ];

	const uint gidx = get_global_id(0);
	const uint lidx = get_local_id (0);

	const uint vector_count = ${ vector_count };

	{
		uint vector_id = 0;
		const size_t visible_size = layer_size(layer);
		const size_t hidden_size = layer_size(layer+1);
		const size_t visible_offset = layer_offset(layer)*vector_count;
		const size_t hidden_offset = layer_offset(layer+1)*vector_count;
		const size_t weights_offsets = weight_offset(layer);

		size_t hidden_id = get_group_id(0);
		for (uint id = hidden_id; id < hidden_size * vector_count; id += get_num_groups(0))
		{
			hidden_id=id%hidden_size;
			vector_id=id/hidden_size;
			float_t val=0.0;

			uint x = lidx;
			for (; x < visible_size; x += get_local_size(0))
			{
				//val += state_arrays[visible_offset + x + visible_size*vector_id] * weight_matrix[weights_offsets + x + (visible_size+1)*hidden_id];

				val += state_arrays[visible_offset + x + visible_size*vector_id] * R_weight_matrix[weights_offsets + x + (visible_size+1)*hidden_id];
				if( layer > 0 )
				{
					float_t i_state = R_state_arrays[visible_offset + x + visible_size*vector_id] * gradient_y(state_arrays[visible_offset + x + visible_size*vector_id]);
					val += i_state * weight_matrix[weights_offsets + x + (visible_size+1)*hidden_id];
				}
			}

			if(x == visible_size)
				val += R_weight_matrix[weights_offsets + visible_size + (visible_size+1)*hidden_id];

			local_store[lidx]=val;

			for (uint stride = get_local_size(0)/2; stride > 0; stride /= 2)
			{
				barrier(CLK_LOCAL_MEM_FENCE);
				if (lidx < stride)
					local_store[lidx] += local_store[lidx + stride];
			}

			if (lidx == 0)
			{		
				R_state_arrays[hidden_offset + id] = (local_store[lidx]);

				if (${layer_count - 2} == layer)
					LJ[id] = R_state_arrays[hidden_offset+id] * state_arrays[hidden_offset+id] * (1.0 - state_arrays[hidden_offset+id]);
			}
		}
	}

}

uint sample_layer(	//returns new offset
	__global float_t *state_arrays,

	const uint layer,

	__global const float_t *random_vector,
	const uint random_size, // power of 2
	uint random_offset
	)
{
	const uint gidx = get_global_id(0);
	const uint lidx = get_local_id (0);

	const size_t vector_count = ${ vector_count };

	const size_t offset = layer_offset(layer)*vector_count;
	const size_t size = layer_size(layer);

	for (uint id = gidx; id < size * vector_count; id += get_global_size(0))
	{
		state_arrays[offset + id] = (state_arrays[offset + id]>random_vector[random_offset])?1.0:0.0;
		random_offset=(random_offset+get_global_size(0))&(random_size-1);
	}
	return random_offset;
}

__attribute__((reqd_work_group_size( ${work_group_size} , 1, 1)))
void rbm_step_back(
	__global const float_t* read,
	__global float_t* write,
	__global const float_t*weights,
	const uint layer
	)
{
	__local ${float_type} local_store[ ${work_group_size} ];
	const uint gidx = get_global_id(0);
	const uint lidx = get_local_id (0);

	uint vector_id = 0;
		//dreaming
	__global const float_t* hid_vector = read;
	__global float_t* neg_vis_vector = write;
	const size_t visible_size = layer_size(layer);
	const size_t hidden_size = layer_size(layer+1);
	const size_t weights_offsets = weight_offset(layer);

	uint visible_id = get_group_id(0);
	for (uint id = visible_id; id < visible_id * ${ vector_count }; id += get_num_groups(0))
	{
		visible_id=id%visible_size;
		vector_id=id/visible_size;

		float_t val=0.0;

		uint x = lidx;

		for (; x < hidden_size; x += get_local_size(0))
		{
			val += hid_vector[x + vector_id*hidden_size] * weights[weights_offsets+visible_id + (hidden_size+1)*x];
		}

		local_store[lidx]=val;

		for (uint stride = get_local_size(0)/2; stride > 0; stride /= 2)
		{
			barrier(CLK_LOCAL_MEM_FENCE);
			if (lidx < stride)
				local_store[lidx] += local_store[lidx + stride];
		}

		if (lidx == 0)
		{		
			neg_vis_vector[id] = sigmoid(local_store[lidx]);
		}

	}
}

__attribute__((reqd_work_group_size( ${work_group_size} , 1, 1)))
void rbm_step_forward(	
	__global const float_t *read,
	__global float_t *write,

	__global const float_t *weight_matrix,

	const uint layer
	)
{
	__local ${float_type} local_store[ ${work_group_size} ];

	const uint gidx = get_global_id(0);
	const uint lidx = get_local_id (0);

	const uint vector_count = ${ vector_count };

	uint vector_id = 0;
	const size_t visible_size = layer_size(layer);
	const size_t hidden_size = layer_size(layer+1);
	const size_t weights_offsets = weight_offset(layer);

	uint hidden_id = get_group_id(0);
	for (uint id = hidden_id; id < hidden_size * vector_count; id += get_num_groups(0))
	{
		hidden_id=id%hidden_size;
		vector_id=id/hidden_size;
		float_t val=0.0;

		uint x = lidx;
		for (; x < visible_size; x += get_local_size(0))
		{
			val += read[x + visible_size*vector_id] * weight_matrix[weights_offsets + x + (visible_size+1)*hidden_id];
		}

		if(x == visible_size)
			val += weight_matrix[weights_offsets + visible_size + (visible_size+1)*hidden_id];

		local_store[lidx]=val;

		for (uint stride = get_local_size(0)/2; stride > 0; stride /= 2)
		{
			barrier(CLK_LOCAL_MEM_FENCE);
			if (lidx < stride)
				local_store[lidx] += local_store[lidx + stride];
		}

		if (lidx == 0)
		{		
			write[id] = sigmoid(local_store[lidx]);
		}
	}
}


// TODO: Currently has race conditions on global memory, and so disabled.
/*
__attribute__((reqd_work_group_size( ${work_group_size} , 1, 1)))
__kernel void contrastive_divergence(	
	__global float_t *state_arrays,
	__global float_t *dreaming,

	__global float_t *weight_matrix,

	__global float_t *delta_matrix,

	__global const float_t *random_vector,
	const uint random_size, // power of 2

	const uint rbm_target,
	const uint sample_steps, // 1 is fine, but we can do more..
	const float_t learning_rate, // 0.1 default
	const uint iterations
	)
{
	uint random_offset = get_global_id(0);

	const uint vector_count = ${vector_count};


	random_offset = sample_layer(state_arrays,rbm_target+1,random_vector,random_size,random_offset);

	__global float_t* neg_hid_vector = &state_arrays[layer_offset(rbm_target+1)*vector_count];
	__global float_t* neg_vis_vector  = dreaming;
	__global float_t* hid_vector = &state_arrays[layer_offset(rbm_target+1)*vector_count];
	__global float_t* vis_vector  = &state_arrays[layer_offset(rbm_target)*vector_count];

	for(uint n = 0; n < sample_steps; n++)
	{
		rbm_step_back(neg_hid_vector,neg_vis_vector,weight_matrix,rbm_target);
		barrier(CLK_GLOBAL_MEM_FENCE);
		neg_hid_vector  = &dreaming[layer_size(rbm_target)*vector_count];
		rbm_step_forward(neg_vis_vector,neg_hid_vector,weight_matrix,rbm_target);
		barrier(CLK_GLOBAL_MEM_FENCE);
		random_offset = sample_layer(state_arrays,rbm_target+1,random_vector,random_size,random_offset);
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	{
		__local ${float_type} local_store[ ${work_group_size} ];

		const uint gidx = get_global_id(0);
		const uint lidx = get_local_id (0);

		uint vector_id = 0;
		const uint visible_size = layer_size(rbm_target);
		const uint hidden_size = layer_size(rbm_target+1);
		const uint weights_offsets = weight_offset(rbm_target);

		for (uint hidden_id = get_group_id(0); hidden_id < hidden_size; hidden_id += get_num_groups(0))
		{
			for (uint visible_id = (0); visible_id <= visible_size; visible_id ++)
			{
				float_t val=0.0;

				uint x = lidx;
				for (; x < vector_count; x += get_local_size(0))
				{
					if(visible_id<visible_size)
						val += vis_vector[visible_id+x*visible_size]*hid_vector[hidden_id+x*hidden_id] - neg_vis_vector[visible_id+x*visible_size]*neg_hid_vector[hidden_id+x*hidden_id];
					else
						val += hid_vector[hidden_id+x*hidden_id] - neg_hid_vector[hidden_id+x*hidden_id];
				}

				local_store[lidx]=val;

				for (uint stride = get_local_size(0)/2; stride > 0; stride /= 2)
				{
					barrier(CLK_LOCAL_MEM_FENCE);
					if (lidx < stride)
						local_store[lidx] += local_store[lidx + stride];
				}

				if (lidx == 0)
				{
					delta_matrix[weights_offsets + visible_id + (visible_size+1)*hidden_id]+= learning_rate*local_store[lidx];
				}
			}
		}
	}
}*/


__attribute__((reqd_work_group_size( ${work_group_size} , 1, 1)))
__kernel void copy_buffers(
	__global const float_t *buf_arrays,
	__global float_t *state_arrays,
	__global float_t *targets
	)
{
	const uint gidx = get_global_id(0);
	for (uint id = get_global_id(0); id < layer_size(0) * ${ vector_count }; id += get_global_size(0))
	{
		uint idx = id%layer_size(0);
		uint vidx = id/layer_size(0);
		state_arrays[id] = buf_arrays[ idx + ( layer_size(0) + layer_size(${ layer_count }-1) )*vidx ];
	}

	for (uint id = get_global_id(0); id < layer_size( ${ layer_count }-1 ) * ${ vector_count }; id += get_global_size(0))
	{
		uint idx = id%layer_size( ${ layer_count }-1 ) ;
		uint vidx = id/layer_size( ${ layer_count }-1 );
		targets[id] = buf_arrays[ layer_size(0) + idx + ( layer_size(0) + layer_size(${ layer_count }-1) )*vidx ];
	}
}

__attribute__((reqd_work_group_size( ${work_group_size} , 1, 1)))
__kernel void load_error(
	__global const float_t *state_arrays,
	__global const float_t *targets,
	__global float_t *error
	)
{
	const uint gidx = get_global_id(0);
	for (uint id = get_global_id(0); id < layer_size(${ layer_count }-1) * ${ vector_count }; id += get_global_size(0))
	{
		//uint idx = id%layer_size(${ layer_count }-1);
		//uint vidx = id/layer_size(${ layer_count }-1);
		error[id] = state_arrays[id] - targets[id];
	}
}


__attribute__((reqd_work_group_size( ${work_group_size} , 1, 1)))
__kernel void backward_pass(	
	__global const float_t *state_arrays,
	__global const float_t *initial_error,

	__global const float_t *weight_matrix,
	__global float_t *result_weights,

	__global float_t *out_error,

	const uint layer
	)
{
	__local ${float_type} local_store[ ${work_group_size} ];

	const uint gidx = get_global_id(0);
	const uint lidx = get_local_id (0);

	const uint vector_count = ${ vector_count };

	const float_t multiplier = 1./vector_count;

	uint vector_id = 0;
	const size_t visible_size = layer_size(layer);
	const size_t hidden_size = layer_size(layer+1);
	const size_t visible_offset = layer_offset(layer)*vector_count;
	const size_t hidden_offset = layer_offset(layer+1)*vector_count;
	const size_t weights_offsets = weight_offset(layer);


		size_t id = get_group_id(0);
		size_t visible_id = 0;
		size_t hidden_id = 0;
		hidden_id=id/(visible_size+1);
		visible_id=id%(visible_size+1);
		while(hidden_id< hidden_size)
		{
			uint x=lidx;

			float_t val=0.0;


			for (vector_id=lidx; vector_id < vector_count; vector_id += get_local_size(0))
			{
				float_t error_v = initial_error[hidden_id + hidden_size*vector_id];

				if(visible_id < visible_size)
					val += (state_arrays[visible_offset+visible_id + visible_size*vector_id] * gradient_y(state_arrays[hidden_offset+hidden_id + hidden_size*vector_id]) )*error_v ;
				else
					val += (gradient_y(state_arrays[hidden_offset+hidden_id + hidden_size*vector_id]) )*error_v;
				//val+=error_v;
			}


			local_store[lidx]=val;

			for (uint stride = get_local_size(0)/2; stride > 0; stride /= 2)
			{
				barrier(CLK_LOCAL_MEM_FENCE);
				if (lidx < stride)
					local_store[lidx] += local_store[lidx + stride];
			}

			if (lidx == 0)
			{
				result_weights[weights_offsets + visible_id + (visible_size+1)*hidden_id] = local_store[lidx]*multiplier;
			}

			id += get_num_groups(0);
			hidden_id=id/(visible_size+1);
			visible_id=id%(visible_size+1);
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if( layer > 0 )
		{
			for(size_t id = get_group_id(0); id < (visible_size )* vector_count; id += get_num_groups(0))
			{
				visible_id=id%visible_size;
				vector_id=id/visible_size;
				float_t val=0.0;

				uint x = lidx;

				for (; x < hidden_size; x += get_local_size(0))
				{
					float_t error_v = initial_error[x + hidden_size*vector_id];

					val += gradient_y(state_arrays[hidden_offset+x + hidden_size*vector_id]) *  weight_matrix[weights_offsets + visible_id + (visible_size+1)*x] * 
					error_v;
				}

				local_store[lidx]=val;

				for (uint stride = get_local_size(0)/2; stride > 0; stride /= 2)
				{
					barrier(CLK_LOCAL_MEM_FENCE);
					if (lidx < stride)
						local_store[lidx] += local_store[lidx + stride];
				}

				if (lidx == 0)
				{		
					out_error[visible_id + visible_size*vector_id] = local_store[lidx];
				}
			}
		}
}
__kernel void zero(	
	__global float_t *input
	)
{
	const uint gidx = get_global_id(0);
	const uint lidx = get_local_id (0);

	input[gidx]=0.;
}

__attribute__((reqd_work_group_size( ${work_group_size} , 1, 1)))
__kernel void calc_loss(
	__global float_t *state_arrays,

	__global const float_t *weight_matrix,
	
	__global float_t *target,
	__global float_t *output
)
{
	__local ${float_type} local_store[ ${work_group_size} ];

	const uint gidx = get_global_id(0);
	const uint lidx = get_local_id (0);
	const float_t multiplier = 1./( ${vector_count} );
	const uint outsize = layer_size( ${layer_count}-1 );

	__global const float_t* input = &state_arrays[  layer_offset( ${layer_count}-1 )*${vector_count} ];

	for (int v = get_group_id(0); v < ${vector_count}; v+=get_num_groups(0))
	{
		float_t val=0.0;

		uint x = lidx;

		for (; x < layer_size( ${layer_count}-1 ); x += get_local_size(0))
		{
			val -= (target[x + v*outsize]*log(input[x + v*outsize]) + (1.0f-target[x + v*outsize])*log1p(-input[x + v*outsize]));
		}

		if(isfinite(val)==0)
			val = 1.0E+20f;

		local_store[lidx]=val;

		for (uint stride = get_local_size(0)/2; stride > 0; stride /= 2)
		{
			barrier(CLK_LOCAL_MEM_FENCE);
			if (lidx < stride)
				local_store[lidx] += local_store[lidx + stride];
		}

		if (lidx == 0)
		{		
			output[v] = local_store[lidx];
		}
	}

}


__attribute__((reqd_work_group_size( ${work_group_size} , 1, 1)))
__kernel void sum(
	__global float_t *vector,
	const uint inv_multiplier,

	const uint size
)
{
	__local ${float_type} local_store[ ${work_group_size} ];

	const uint gidx = get_global_id(0);
	const uint lidx = get_local_id (0);

	const float_t multiplier = 1./inv_multiplier;
	if(0 == get_group_id(0))
	{
		float_t val=0.;

		uint x = lidx;

		for (; x < ( ${vector_count} ); x += get_local_size(0))
		{
			val += vector[x];
		}

		local_store[lidx]=val;

		for (uint stride = get_local_size(0)/2; stride > 0; stride /= 2)
		{
			barrier(CLK_LOCAL_MEM_FENCE);
			if (lidx < stride)
				local_store[lidx] += local_store[lidx + stride];
		}

		if (lidx == 0)
		{		
			vector[0] = local_store[lidx]*multiplier;
		}
	}
}