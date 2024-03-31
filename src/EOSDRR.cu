#include "EOSDRR.h"
#include "cutil_math.h"
#include <stdio.h>
#include <texture_types.h>
__global__ void EOSKernel(unsigned short *img, cudaTextureObject_t tex,
                          const float *world_to_model, const unsigned int z0,
                          const float lambda, const float lambda_z,
                          const float sid, const float sdd,
                          const unsigned int C, const unsigned int R,
                          const unsigned int is_lateral, const float cutoff,
                          const float intensity) {
  // Image is of size width x height
  const int i = blockIdx.x * blockDim.x + threadIdx.x; // Col
  const int j = blockIdx.y * blockDim.y * threadIdx.y; // Row

  if (i >= C || j >= R)
    return;

  float z = z0 - lambda_z * j; // Height index (world coordinates)
  printf("Z: %f\n", z);

  // Emitter Position [Ray Origin] (world coordinates)
  float3 e = make_float3(-sid, 0.f, z);
  if (is_lateral) {
    e = make_float3(0.f, -sid, z);
  }

  // Detector position (world coordinates)
  float3 d = make_float3(sdd - sid, ((i - (C / 2.f)) * lambda * sdd) / sid, z);
  if (is_lateral) {
    d = make_float3(((i - (C / 2.f)) * lambda * sdd) / sid, sdd - sid, z);
  }

  // Ray direction
  float3 r = d - e;
  r = normalize(r);

  // Ray marching back to front
  float density = 0.f;
  float t = sdd;
  float near = 0.f;
  float step = 0.1f;
  while (t > near) {
    float3 p = e + t * r;
    // To Model coordinates
    // Rotate
    p = make_float3(dot(make_float3(world_to_model[0], world_to_model[4],
                                    world_to_model[8]),
                        p),
                    dot(make_float3(world_to_model[1], world_to_model[5],
                                    world_to_model[9]),
                        p),
                    dot(make_float3(world_to_model[2], world_to_model[6],
                                    world_to_model[10]),
                        p));
    // Translate
    p +=
        make_float3(world_to_model[12], world_to_model[13], world_to_model[14]);
    // Sample
    unsigned short sample = tex3D<unsigned short>(tex, p.x, -p.y, -p.z);
    if (sample > cutoff) {
      density += sample;
    }
    t -= step;
  }

  // write to Image
  if (density > 0.1) {
    printf("Density: %f, j: %i, i: %i\n", density, j, i);
    img[j * C + i] = density / intensity;
  }
}

void eos_projection(unsigned short *img, cudaTextureObject_t tex,
                    const float *world_to_model, const unsigned int z0,
                    const float lambda, const float lambda_z, const float sid,
                    const float sdd, const unsigned int C, const unsigned int R,
                    const unsigned int is_lateral, const float cutoff,
                    const float intensity) {
  dim3 dimGrid(ceil(C / 16.0), ceil(R / 16), 1);
  dim3 dimBlock(16, 16, 1);
  EOSKernel<<<dimGrid, dimBlock>>>(img, tex, world_to_model, z0, lambda,
                                   lambda_z, sid, sdd, C, R, is_lateral, cutoff,
                                   intensity);
}
