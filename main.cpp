#include "src/EOSDRR.h"
#include "src/TiffImage.h"
#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_functions.h>
#include <fstream>
#include <iostream>
#include <string>
#include <texture_types.h>

// #define DEBUG 1

void read_tiff(const std::string &fName, unsigned short *data) {
  TIFF *tif = TIFFOpen(fName.c_str(), "r");
  if (!tif) {
    std::cout << "Error opening TIFF file" << std::endl;
    exit(1);
  }

  TiffImage img;
  tiffImageReadMeta(tif, &img);

  int dircount = 0;
  do {
    dircount++;
  } while (TIFFReadDirectory(tif));

  int width = img.width;
  int height = img.height;
  int depth = dircount;
  int bps = img.bitsPerSample;

  unsigned short *dp = data;
  for (int i = 0; i < depth; i++) {
    TIFFSetDirectory(tif, i);
    tiffImageRead(tif, &img);
    if (img.width != width || img.height != height) {
      std::cout << "Error: image dimensions do not match" << std::endl;
      exit(1);
    }
    memcpy(dp, img.data, width * height * bps / 8);
    tiffImageFree(&img);
    dp += width * height * bps / 16;
  }

  TIFFClose(tif);
}

void save_debug_image(const std::string &fName, unsigned short *data, int width,
                      int height) {
  // Write out a pgm file for debugging
  std::ofstream ofs(fName.c_str(), std::ios::binary);
  ofs << "P5\n" << width << " " << height << "\n" << 65535 << "\n";
  ofs.write((char *)data, width * height * sizeof(unsigned short));
  ofs.close();
}

inline void checkCudaError(cudaError_t err, int line_num) {
  if (err != cudaSuccess) {
    std::cout << (err) << ": " << cudaGetErrorString(err) << " in " << __FILE__
              << " at " << line_num << std::endl;
    exit(EXIT_FAILURE);
  }
}

// EOS parameters
float d_f = 1300.f;         // Frontal emitter to detector distance (mm)
float f_f = 987.f;          // Frontal emitter to isocenter distance (mm)
float d_l = 1300.f;         // Lateral emitter to detector distance (mm)
float f_l = 918.f;          // Lateral emitter to isocenter distance (mm)
float w_f = 450.f;          // Frontal detector width (mm)
float w_l = 450.f;          // Lateral detector width (mm)
float lambda_f = 0.179363f; // Frontal detector pixel pitch (mm)
float lambda_l = 0.179363f; // Lateral detector pixel pitch (mm)
float lambda_z = 0.179363f; // Vertical pixel pitch (mm)
unsigned int R = 3000;      // Number of rows
unsigned int C_f =
    1896; // Highest column index in frontal DRR (C_f + 1 total columns)
unsigned int C_l =
    1763;       // Highest column index in lateral DRR (C_l + 1 total columns)
float z0 = 0.f; // Initial z value (mm)
unsigned int is_frontal = 0; // 1 if frontal, 0 if lateral

int main(int argc, char *argv[]) {
#ifdef DEBUG
  std::cout << "Df " << d_f << std::endl;
  std::cout << "Ff " << f_f << std::endl;
  std::cout << "Dl " << d_l << std::endl;
  std::cout << "Fl " << f_l << std::endl;
  std::cout << "Wf " << w_f << std::endl;
  std::cout << "Wl " << w_l << std::endl;
  std::cout << "Lf " << lambda_f << std::endl;
  std::cout << "Ll " << lambda_l << std::endl;
  std::cout << "Lz " << lambda_z << std::endl;
  std::cout << "R " << R << std::endl;
  std::cout << "Cf " << C_f << std::endl;
  std::cout << "Cl " << C_l << std::endl;
#endif

  int vol_dims[3] = {60, 59, 116};
  size_t size = 60 * 59 * 116 * sizeof(unsigned short);

  unsigned short *vol_data =
      new unsigned short[vol_dims[0] * vol_dims[1] * vol_dims[2] * 2];
  read_tiff("/home/aj/Documents/EOS-Test-Bed/Volumes/mc3_dcm_cropped.tif",
            vol_data);

  float world_to_vol[16] = {0.0f};
  world_to_vol[0] = 1.f / 0.390625f;
  world_to_vol[5] = 1.f / 0.390625f;
  world_to_vol[10] = 1.f / 0.625f;
  world_to_vol[12] = -50.f; // X translation
  world_to_vol[13] = 50.f;  // Y translation
  world_to_vol[14] = 50.f;  // Z translation
  world_to_vol[15] = 1.0f;

  // Create Image and volume on GPU
  unsigned short *out;
  checkCudaError(cudaMalloc((void **)&out, C_f * R * sizeof(unsigned short)),
                 __LINE__);

  // Allocate space on gpu for volume
  cudaChannelFormatDesc desc =
      cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);
  cudaArray_t arr;
  cudaExtent extent = make_cudaExtent(60, 59, 116);
  checkCudaError(cudaMalloc3DArray(&arr, &desc, extent), __LINE__);

  // Copy volume to gpu
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr = make_cudaPitchedPtr(&vol_data[0], extent.width * 2,
                                          extent.width, extent.height);
  copyParams.dstArray = arr;
  copyParams.extent = extent;
  copyParams.kind = cudaMemcpyHostToDevice;

  checkCudaError(cudaMemcpy3D(&copyParams), __LINE__);

  // Set up textureObject
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = arr;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.normalizedCoords = true;
  // texDesc.filterMode = cudaFilterModeLinear;
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.readMode = cudaReadModeElementType;

  cudaTextureObject_t tex = 0;
  checkCudaError(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL),
                 __LINE__);

  // Create world_to_vol mat on GPU
  float *wtv;
  checkCudaError(cudaMalloc((void **)&wtv, 16 * sizeof(float)), __LINE__);
  // Copy to GPU
  checkCudaError(
      cudaMemcpy(wtv, world_to_vol, 16 * sizeof(float), cudaMemcpyHostToDevice),
      __LINE__);

  // Kernel
  int is_lateral = 0;
  float cutoff = 0.0f;
  float intensity = 0.62f;
  eos_projection(out, tex, wtv, z0, lambda_f, lambda_z, f_f, d_f, C_f, R,
                 is_lateral, cutoff, intensity);
  checkCudaError(cudaGetLastError(), __LINE__);

  // Pull image off of GPU
  unsigned short *img = new unsigned short[C_f * R * sizeof(unsigned short)];
  checkCudaError(cudaMemcpy(img, out, C_f * R * sizeof(unsigned short),
                            cudaMemcpyDeviceToHost),
                 __LINE__);
  save_debug_image("test-cuda.pgm", img, C_f, R);

  // Clean up
  free(vol_data);

  checkCudaError(cudaFree(out), __LINE__);
  checkCudaError(cudaFree(wtv), __LINE__);

  return 0;
}
