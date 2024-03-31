#include <texture_types.h>
void eos_projection(unsigned short *img, cudaTextureObject_t tex,
                    const float *world_to_model, const unsigned int z0,
                    const float lambda, const float lambda_z, const float sid,
                    const float sdd, const unsigned int C, const unsigned int R,
                    const unsigned int is_lateral, const float cutoff,
                    const float intensity);
