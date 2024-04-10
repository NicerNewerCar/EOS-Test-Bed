#include "src/Raycaster.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
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

void cv_write(float *buffer, int R, int C, std::string filename) {
  cv::Mat color;
  color = cv::Mat::zeros(R, C, CV_64F);

  // Fill buffer
  for (int v = 0; v < R; ++v) {
    for (int u = 0; u < C; ++u) {
      float val = buffer[R * u + v];
      color.at<double>(v, u) = val;
    }
  }

  // Scale to 0-255
  double min;
  double max;
  cv::minMaxIdx(color, &min, &max);
  float scale = 255 / (max - min);
  cv::Mat color_scaled;
  color_scaled = cv::Mat::zeros(R, C_f, CV_8UC1);
  color.convertTo(color_scaled, CV_8UC1, scale, -min * scale);

  cv::imwrite(filename, color_scaled);
}

int main(int argc, const char **argv) {
  std::string filename =
      "/home/aj/Documents/EOS-Test-Bed/Volumes/eos-test.nrrd";

  RayCaster frontal_caster(d_f, f_f, lambda_f, lambda_z, false);
  frontal_caster.setVolume(filename);

  float *buffer = new float[C_f * R * sizeof(float)];
  frontal_caster.render(buffer, C_f, R, z0);
  cv_write(buffer, R, C_f, "EOS-frontal-test.pgm");

  RayCaster lateral_caster(d_l, f_l, lambda_l, lambda_z, true);
  lateral_caster.setVolume(filename);

  free(buffer);
  buffer = new float[C_l * R * sizeof(float)];
  lateral_caster.render(buffer, C_l, R, z0);
  cv_write(buffer, R, C_l, "EOS-lateral-test.pgm");

  return 0;
}
