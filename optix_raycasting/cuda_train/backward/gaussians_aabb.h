#pragma once
#define BUFFER_SIZE 32
#define CHUNK_SIZE 16
#define TRANSMITTANCE_EPSILON 0.001f
#define SIGMA_THRESHOLD 0.1f

struct alignas(16) PrimData {
    float4 pos4;       // positions_relative4 origin-position
    float4 inv_inter4; // inv_scales_intersect4   inv_scale/sqrt(2.0*log(sig/sig_thresh))
    float4 quat4;      // quaternions
    float4 rgba4;      // rgbsigma, rgb is pre-multiplied by sigma
  };

struct alignas(16) Params
{   
    unsigned int rnd_sample;
    unsigned int           max_prim_slice;

    float3 bbox_min;
    float3 bbox_max;

    float dt_step;
    unsigned int dynamic_sampling;

    unsigned int image_width;
    unsigned int image_height;

    float3 cam_eye;
    float3 cam_u;
    float3 cam_v;
    float3 cam_w;

    float4 cam_intr; // {tan_half_fovx, tan_half_fovy, principal_point_x, principal_point_y}

    const PrimData* __restrict__ prims;

    int* __restrict__ hit_prim_idx; // index of the primitive that was hit

    float3* __restrict__ ray_colors;
    float3* __restrict__ dloss_dray_colors;
    float* __restrict__ densities_grad;
    float3* __restrict__ color_features_grad;
    float3* __restrict__ positions_grad;
    float3* __restrict__ scales_grad;
    float4* __restrict__ quaternions_grad;

    OptixTraversableHandle handle;
};


struct MissData
{
    float3 bg_color;
};


template<unsigned int N>
static __host__ __device__ __inline__ unsigned int tea( unsigned int val0, unsigned int val1 )
{
  unsigned int v0 = val0;
  unsigned int v1 = val1;
  unsigned int s0 = 0;

  for( unsigned int n = 0; n < N; n++ )
  {
    s0 += 0x9e3779b9;
    v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
    v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
  }

  return v0;
}


// Generate random unsigned int in [0, 2^24)
static __host__ __device__ __inline__ unsigned int lcg(unsigned int &prev)
{
  const unsigned int LCG_A = 1664525u;
  const unsigned int LCG_C = 1013904223u;
  prev = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd(unsigned int &prev)
{
  return ((float) lcg(prev) / (float) 0x01000000);
}
