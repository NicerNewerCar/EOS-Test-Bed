#include "src/Raycaster.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <string>

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
unsigned int C_f = 1896;
// Highest column index in frontal DRR (C_f + 1 total columns)
unsigned int C_l = 1763;
// Highest column index in lateral DRR (C_l + 1 total columns)
float z0 = 100.f;            // Initial z value (mm)
unsigned int is_lateral = 0; // 0 if frontal, 1 if lateral

void rescale_write(float *buffer, int R, int C, std::string filename) {
  float old_min = FLT_MAX;
  float old_max = FLT_MIN;
  for (int v = 0; v < R; ++v) {
    for (int u = 0; u < C; ++u) {
      float val = buffer[C * v + u];
      old_min = (val < old_min) ? val : old_min;
      old_max = (val > old_max) ? val : old_max;
    }
  }
  unsigned char *scaled_buffer =
      new unsigned char[C * R * sizeof(unsigned char)];
  unsigned char new_min = 0;
  unsigned char new_max = 255;
  for (int v = 0; v < R; ++v) {
    for (int u = 0; u < C; ++u) {
      float val = buffer[C * v + u];
      unsigned char scaled_val =
          ((val - old_min) / (old_max - old_min)) * (new_max - new_min) +
          new_min;
      scaled_buffer[C * v + u] = scaled_val;
    }
  }

  std::ofstream ofs(filename.c_str(), std::ios::out);
  ofs << "P5\n" << C << " " << R << "\n255\n";
  ofs.write((char *)scaled_buffer, C * R * sizeof(unsigned char));
  ofs.close();
}

int main(int argc, const char **argv) {
  std::string filename =
      "/home/aj/Documents/EOS-Test-Bed/Volumes/eos-test.nrrd";

  Eigen::Matrix3f rotation_x, rotation_y, rotation_z, rotation;
  rotation_x = Eigen::Matrix3f::Identity();
  rotation_y = Eigen::Matrix3f::Identity();
  rotation_z = Eigen::Matrix3f::Identity();

  RayCaster frontal_caster(d_f, f_f, lambda_f, lambda_z, false);
  frontal_caster.setVolume(filename);

  float *buffer = new float[C_f * R * sizeof(float)];
  frontal_caster.render(buffer, C_f, R, z0);
  rescale_write(buffer, R, C_f, "EOS-frontal-test.pgm");

  RayCaster lateral_caster(d_l, f_l, lambda_l, lambda_z, true);
  lateral_caster.setVolume(filename);

  free(buffer);
  buffer = new float[C_l * R * sizeof(float)];
  lateral_caster.render(buffer, C_l, R, z0);
  rescale_write(buffer, R, C_l, "EOS-lateral-test.pgm");

  return 0;
}
