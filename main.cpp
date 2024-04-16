#include "src/Raycaster.hpp"
#include <cfloat>
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
unsigned int R = 1500;      // 3000;      // Number of rows
unsigned int C_f = 1500;    // 1896;
// Highest column index in frontal DRR (C_f + 1 total columns)
unsigned int C_l = 1500; // 1763;
// Highest column index in lateral DRR (C_l + 1 total columns)
float z0 = 150.f; // Initial z value (mm)

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
      scaled_buffer[C * v + u] = 255 - scaled_val;
    }
  }

  std::ofstream ofs(filename.c_str(), std::ios::out);
  ofs << "P5\n" << C << " " << R << "\n255\n";
  ofs.write((char *)scaled_buffer, C * R * sizeof(unsigned char));
  ofs.close();
  free(scaled_buffer);
}

void render_write(RayCaster *caster, int R, int C, std::string filename) {
  float *buffer = new float[C * R * sizeof(float)];
  caster->render(buffer, C, R, z0);
  rescale_write(buffer, R, C, filename);
  free(buffer);
}

int main(int argc, const char **argv) {
  std::string filename =
      "/home/aj/Documents/EOS-Test-Bed/Volumes/eos-test.nrrd";

  RayCaster frontal_caster(d_f, f_f, lambda_f, lambda_z, false);
  frontal_caster.setVolume(filename);
  render_write(&frontal_caster, R, C_f, "EOS-frontal-test.pgm");

  RayCaster lateral_caster(d_l, f_l, lambda_l, lambda_z, true);
  lateral_caster.setVolume(filename);
  render_write(&lateral_caster, R, C_l, "EOS-lateral-test.pgm");
  return 0;
}
