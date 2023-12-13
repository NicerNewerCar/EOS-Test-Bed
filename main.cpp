#include <iostream>
#include <fstream>
#include <string>
#include <windows.h>
#include <CL/opencl.h>
#include "src/TiffImage.h"



#define DEBUG 1

// EOS parameters
float d_f = 1300.f; // Frontal emitter to detector distance (mm)
float f_f = 987.f; // Frontal emitter to isocenter distance (mm)
float d_l = 1300.f; // Lateral emitter to detector distance (mm)
float f_l = 918.f; // Lateral emitter to isocenter distance (mm)
float w_f = 450.f; // Frontal detector width (mm)
float w_l = 450.f; // Lateral detector width (mm)
float lambda_f = 0.179363f; // Frontal detector pixel pitch (mm)
float lambda_l = 0.179363f; // Lateral detector pixel pitch (mm)
float lambda_z = 0.179363f; // Vertical pixel pitch (mm)
unsigned int R = 3000; // Number of rows
unsigned int C_f = 1896; // Highest column index in frontal DRR (C_f + 1 total columns)
unsigned int C_l = 1763; // Highest column index in lateral DRR (C_l + 1 total columns)
float z0 = 0.f; // Initial z value (mm)
unsigned int is_frontal = 0; // 1 if frontal, 0 if lateral


void check_error(cl_int err, const char* msg) {
  if (err != CL_SUCCESS) {
    std::cout << msg <<  " " << err <<std::endl;
    throw std::runtime_error(msg);
  }
}

void read_tiff(const std::string& fName, unsigned short* data) {
  TIFF* tif = TIFFOpen(fName.c_str(), "r");
  if (!tif) {
    std::cout << "Error opening TIFF file" << std::endl;
    exit(1);
  }

  TiffImage img;
  tiffImageReadMeta(tif, &img);
  
  int dircount = 0;
  do {
      dircount++;
  } while(TIFFReadDirectory(tif));

  int width = img.width;
  int height = img.height;
  int depth = dircount;
  int bps = img.bitsPerSample;

  unsigned short* dp = data;
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

void save_debug_image(const std::string& fName, unsigned short* data, int width, int height) {
  // Write out a pgm file for debugging
  std::ofstream ofs(fName.c_str(), std::ios::binary);
  ofs << "P5\n" << width << " " << height << "\n" << 65535 << "\n";
  ofs.write((char*)data, width * height * sizeof(unsigned short));
  ofs.close();
}

int main(int argc, char* argv[]) {

  unsigned int is_lateral = 0;
  unsigned int vol_dims[3];
  std::string volName;
  std::string outFname;

  if (argc < 7) {
    std::cerr << "Usage: " << argv[0] << " <volume name> <output name (.pgm)> <is_lateral> <vol_dim_x> <vol_dim_y> <vol_dim_z>" << std::endl;
    return -1;
  }

  volName = argv[1];
  outFname = argv[2];
  is_lateral = atoi(argv[3]);
  vol_dims[0] = atoi(argv[4]);
  vol_dims[1] = atoi(argv[5]);
  vol_dims[2] = atoi(argv[6]);

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

  // Set up OpenCL
  cl_int err;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;
  cl_mem input_vol;
  cl_mem image;
  cl_mem output_drr;
  cl_mem world_to_vol_buf;
  cl_mem viewport_buf;

  // Get platform
  cl_uint num_platforms;
  cl_platform_id platforms[10];
  check_error(clGetPlatformIDs(10, platforms, &num_platforms),"Error getting platforms");
  //platform = platforms[0]; // This is the GPU - Use this for OclGrind too
  platform = platforms[1]; // This is the Integreated Graphics
  //platform = platforms[2]; // This is the CPU

  // Get device
  check_error(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL), "Error getting device");

#ifdef DEBUG
// Print out device name
  char device_name[128];
  check_error(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL), "Error getting device name");
  std::cout << "Device: " << device_name << std::endl;
#endif

  //Context properties
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};

  // Create context
  context = clCreateContext(properties, 1, &device, NULL, NULL, &err);
  check_error(err, "Error creating context");

  // Create command queue
  queue = clCreateCommandQueue(context, device, 0, &err);
  check_error(err, "Error creating command queue");

  // Create program
  std::string kernel_source = "C:/Users/anthony.lombardi/Projects/EOS/src/EosDrr2D.cl";
  std::ifstream kernel_file(kernel_source);
  if (!kernel_file.is_open()) {
    std::cout << "Error opening kernel file" << std::endl;
    exit(1);
  }
  std::string kernel_code(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));
  const char* kernel_source_cstr = kernel_code.c_str();
  program = clCreateProgramWithSource(context, 1, &kernel_source_cstr, NULL, &err);
  check_error(err, "Error creating program");

  // Build program
  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char* log = new char[log_size];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    std::cout << log << std::endl;
    delete[] log;
    exit(1);
  }
  check_error(err,"Error building program");

  // Create kernel
  kernel = clCreateKernel(program, "eos_project_drr", &err);
  check_error(err, "Error creating kernel");

  // Create input volume as a
  unsigned short* vol_data = new unsigned short[vol_dims[0] * vol_dims[1] * vol_dims[2] * 2];
  read_tiff(volName, vol_data);
  input_vol = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vol_dims[0] * vol_dims[1] * vol_dims[2] * 2, vol_data, &err);
  check_error(err, "Error creating input volume");
  cl_image_desc vol_desc;
  vol_desc.image_type = CL_MEM_OBJECT_IMAGE3D;
  vol_desc.image_width = vol_dims[0];
  vol_desc.image_height = vol_dims[1];
  vol_desc.image_depth = vol_dims[2];
  vol_desc.image_array_size = 0;
  vol_desc.image_row_pitch = 0;
  vol_desc.image_slice_pitch = 0;
  vol_desc.num_mip_levels = 0;
  vol_desc.num_samples = 0;
  vol_desc.buffer = NULL;
  cl_image_format vol_format;
  vol_format.image_channel_order = CL_R;
  vol_format.image_channel_data_type = CL_UNSIGNED_INT16;

  image = clCreateImage(context, CL_MEM_READ_ONLY, &vol_format, &vol_desc, NULL, &err);
  check_error(err, "Error creating input volume image");
  size_t origin[] = {0, 0, 0};
  size_t dims[] = {vol_dims[0], vol_dims[1], vol_dims[2]};
  err = clEnqueueWriteImage(queue, image, CL_TRUE, origin, dims, 0, 0, vol_data, 0, NULL, NULL);
  check_error(err, "Error writing input volume image");

  float world_to_vol[16] = { 0.0f };
  world_to_vol[0] = 1.f / 0.390625f;
  world_to_vol[5] = 1.f / 0.390625f;
  world_to_vol[10] = 1.f / 0.625f;
  world_to_vol[12] = -50.f; // X translation
  world_to_vol[13] = 50.f; // Y translation
  world_to_vol[14] = 50.f; // Z translation
  world_to_vol[15] = 1.0f;

  world_to_vol_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 16 * sizeof(float), world_to_vol, &err);
  check_error(err, "Error creating world to volume transform");

  // Find the corners of the volume in world space
  float vol_corners[8][4] = { 0.0f };
  vol_corners[0][3] = 1.f;
  vol_corners[1][0] = vol_dims[0] - 1.f;
  vol_corners[1][3] = 1.f;
  vol_corners[2][1] = vol_dims[1] - 1.f;
  vol_corners[2][3] = 1.f;
  vol_corners[3][0] = vol_dims[0] - 1.f;
  vol_corners[3][1] = vol_dims[1] - 1.f;
  vol_corners[3][3] = 1.f;
  vol_corners[4][2] = vol_dims[2] - 1.f;
  vol_corners[4][3] = 1.f;
  vol_corners[5][0] = vol_dims[0] - 1.f;
  vol_corners[5][2] = vol_dims[2] - 1.f;
  vol_corners[5][3] = 1.f;
  vol_corners[6][1] = vol_dims[1] - 1.f;
  vol_corners[6][2] = vol_dims[2] - 1.f;
  vol_corners[6][3] = 1.f;
  vol_corners[7][0] = vol_dims[0] - 1.f;
  vol_corners[7][1] = vol_dims[1] - 1.f;
  vol_corners[7][2] = vol_dims[2] - 1.f;
  vol_corners[7][3] = 1.f;


  // Project the corners onto the DRR image
  //float min_max[4] = { C_f,R,0,0 }; // min_x min_y max_x max_y
  //for (int i = 0; i < 8; i++) {
  //  float world[4] = { 0.0f };
  //  for (int j = 0; j < 4; j++) {
  //    world[j] = vol_corners[i][j];
  //  }
  //  float v = (z0 + world[2]) / lambda_z;
  //  float u_prime = world[1] * f_f;
  //  float w = f_f + world[0];
  //  float u = (C_f / 2.f) + (u_prime) / (lambda_f * w);

  //  // Make sure that the projected point is within the DRR image
  //  if (u > C_f)
  //    u = C_f;
  //  if (u < 0)
  //    u = 0;
  //  if (v > R)
  //    v = R;
  //  if (v < 0)
  //    v = 0;

  //  // Update the min and max x and y values
  //  if (u < min_max[0])
  //    min_max[0] = u;
  //  if (u > min_max[2])
  //    min_max[2] = u;
  //  if (v < min_max[1])
  //    min_max[1] = v;
  //  if (v > min_max[3])
  //    min_max[3] = v;
  //}

  //min_max[2] = min_max[2] - min_max[0];
  //min_max[3] = min_max[3] - min_max[1];

  // Viewport clipping is broken ATM
  float min_max[4] = {0,0, C_f, R};

  // Create the viewport buffer
  viewport_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4 * sizeof(float), min_max, &err);
  unsigned int width = (unsigned int)min_max[2];
  unsigned int height = (unsigned int)min_max[3];


  // Create output DRR C_f * R
  unsigned int drr_dims[] = {width, height};
  output_drr = clCreateBuffer(context, CL_MEM_READ_WRITE, drr_dims[0] * drr_dims[1] * sizeof(unsigned short), NULL, &err);
  check_error(err, "Error creating output DRR");
  // Fill output DRR with zeros
  unsigned short zero = 0;
  err = clEnqueueFillBuffer(queue, output_drr, &zero, sizeof(unsigned short), 0, drr_dims[0] * drr_dims[1] * sizeof(unsigned short), 0, NULL, NULL);
  check_error(err, "Error filling output DRR with zeros");

  

  // Set kernel arguments
  // Projected Image
  static unsigned int arg = 0;
  err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &output_drr);
  check_error(err, "Error setting Projected Image");
  // Volume Image
  err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &image);
  check_error(err, "Error setting Volume");
  // World to Volume Transform
  err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &world_to_vol_buf);
  check_error(err, "Error setting World to Volume Transform");
  // Viewport
  err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &viewport_buf);
  check_error(err, "Error setting Viewport");
  // Z0 (inital emiter position)
  err = clSetKernelArg(kernel, arg++, sizeof(float), &z0);
  check_error(err, "Error setting Z0");
   //Lambda (emiter spacing)
  err = clSetKernelArg(kernel, arg++, sizeof(float), &lambda_f);
  check_error(err, "Error setting lambda");
  // Lambda_z
  err = clSetKernelArg(kernel, arg++, sizeof(float), &lambda_z);
  check_error(err, "Error setting lambda_z");
  // SID (Source to Isocenter Distance)
  err = clSetKernelArg(kernel, arg++, sizeof(float), &f_f);
  check_error(err, "Error setting SID");
  // SDD (Source to Detector Distance)
  err = clSetKernelArg(kernel, arg++, sizeof(float), &d_f);
  check_error(err, "Error setting SDD");
  // C (Number of columns in the detector)
  err = clSetKernelArg(kernel, arg++, sizeof(unsigned int), &C_f);
  check_error(err, "Error setting C");
  // R (Number of rows in the detector)
  err = clSetKernelArg(kernel, arg++, sizeof(unsigned int), &R);
  check_error(err, "Error setting R");
  // is_lateral (0 = frontal view, 1 = lateral view)
  err = clSetKernelArg(kernel, arg++, sizeof(unsigned int), &is_lateral);
  check_error(err, "Error setting is_lateral");
  // cutoff
  float cutoff = 0.0f;
  err = clSetKernelArg(kernel, arg++, sizeof(float), &cutoff);
  check_error(err, "Error setting cutoff");
  // Intensity
  float intensity = .62f;
  err = clSetKernelArg(kernel, arg++, sizeof(float), &intensity);
  check_error(err, "Error setting intensity");
  // Width
  err = clSetKernelArg(kernel, arg++, sizeof(unsigned int), &width);
  check_error(err, "Error setting width");
  // Height
  err = clSetKernelArg(kernel, arg++, sizeof(unsigned int), &height);
  check_error(err, "Error setting height");


  // Run kernel as a 2D NDRange kernel
  size_t global_work_size[2] = {width, height};
  size_t local_work_size[2] = {1, 500};
  std::cout << "Global work size: " << global_work_size[0] << " x " << global_work_size[1] << std::endl;
  std::cout << "Local work size: " << local_work_size[0] << " x " << local_work_size[1] << std::endl;
  std::cout << "Number of work groups: " << global_work_size[0] / local_work_size[0] << " x " << global_work_size[1] / local_work_size[1] << std::endl;
  std::cout << "Running kernel... " << std::endl;
  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  check_error(err, "Error running kernel");

  //wait for kernel to finish
  check_error(clFinish(queue),"Error waiting for the kernel to finish");
  std::cout << "Done" << std::endl;

  // Copy the output DRR to file
  unsigned short* drr_data = new unsigned short[drr_dims[0] * drr_dims[1]];
  err = clEnqueueReadBuffer(queue, output_drr, CL_TRUE, 0, drr_dims[0] * drr_dims[1] * sizeof(unsigned short), drr_data, 0, NULL, NULL);
  check_error(err, "Error reading output DRR from device");
  save_debug_image(outFname, drr_data, drr_dims[0], drr_dims[1]);
  
  std::cout << "Cleaning up..." << std::endl;
  // Clean up
  delete[] vol_data;
  delete[] drr_data;
  clReleaseMemObject(input_vol);
  clReleaseMemObject(output_drr);
  clReleaseMemObject(world_to_vol_buf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  return 0;
} 


