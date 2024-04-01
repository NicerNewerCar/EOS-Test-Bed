__kernel void eos_project_drr(
  __global float* img,
  __read_only image3d_t image,
  __constant float* world_to_model_mat,
  __constant float* viewport,
  const unsigned int z0,
  const float lambda,
  const float lambda_z,
  const float sid,
  const float sdd,
  const unsigned int C,
  const unsigned int R,
  const unsigned int is_lateral,
  const float cutoff,
  const float intensity,
  const unsigned int width,
  const unsigned int height
  ) {
  // Image is of size width x height
  const int i = get_global_id(1); // Column index in final image
  const int j = get_global_id(0); // Row index in final image
    

  if (i >= width || j >= height) return;

  // Full DRR is of size C x R
  const unsigned int u = (unsigned int)(viewport[0] + i); // Column index in volume
  const unsigned int v = (unsigned int)(viewport[1] + j); // Row index in volume

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | // Natural coordinates
    CLK_ADDRESS_CLAMP | // Clamp to zeros at borders
    CLK_FILTER_NEAREST;//LINEAR; // No interpolation

  float z = z0 - lambda_z * v; // height index in volume (world coordinates)


  // Emiter position (world coordinates) Ray origin
  float3 e = (float3)(-sid, 0.f, z);
  if (is_lateral) {
    e = (float3)(0.f, -sid, z);
  }

  // Detector position (world coordinates)
  float3 d = (float3)(sdd - sid, ((u - (C / 2.f)) * lambda * sdd) / sid, z);
  if (is_lateral) {
    d = (float3)(((C/2.f - u) * lambda * sdd) / sid, sdd - sid, z);
  }

  // Ray direction
  float3 r = normalize(d - e);

  
  // Ray marching back to front
  float density = 0.f;
  float t = sdd;
  float near = 0.f;
  float step = 0.1f;
  while (t > near) {
    float3 p = e + t * r; // Point on ray
    // Transform to model coordinates
    p = (float3)(dot((float3)(world_to_model_mat[0],world_to_model_mat[4],world_to_model_mat[8]), p), 
      dot((float3)(world_to_model_mat[1],world_to_model_mat[5],world_to_model_mat[9]), p), 
      dot((float3)(world_to_model_mat[2],world_to_model_mat[6],world_to_model_mat[10]), p));
    p = p + (float3)(world_to_model_mat[12],world_to_model_mat[13],world_to_model_mat[14]); // Translate
    float value = read_imagef(image, sampler, (float4)(p.x, -p.y, -p.z, 0.f)).x; // Sample volume
    if (value > cutoff) {
      density += value;
    }
    t -= step;
  }


  // Write to image
  if (density > 0) {
    img[j * C + i] = density/intensity;
  }

}
