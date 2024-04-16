#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>

class RayCaster {
public:
  RayCaster()
      : sdd(1300.f), sid(900.f), lamdba_hori(1.f), lamdba_vert(1.f),
        is_lateral(0) {}
  RayCaster(float sdd_, float sid_, float lamdba_hori_, float lamdba_vert_,
            bool is_lateral_)
      : sdd(sdd_), sid(sid_), lamdba_hori(lamdba_hori_),
        lamdba_vert(lamdba_vert_), is_lateral(is_lateral_) {}
  void setVolume(std::string filename) { vol = Vol(filename); }
  void render(float *buffer, int C, int R, int z0);
  float trilinear_interpolation(Eigen::Vector3f p);
  void rayBoxIntersection(Eigen::Vector3f ray_origin,
                          Eigen::Vector3f ray_direction,
                          Eigen::Vector3f s_transformed, float &near,
                          float &far);

  void rotate(float theta, Eigen::Vector3f axis) {
    vol.transformation.rotate(Eigen::AngleAxisf(theta * M_PI / 180.0, axis));
  }
  void translate(Eigen::Vector3f translation) {
    vol.transformation.translate(translation);
  }

private:
  float sdd;
  float sid;
  float lamdba_hori;
  float lamdba_vert;
  bool is_lateral;

  // Simple data struct to allow for easy access of NRRD volumes
  struct Vol {
    Vol() {}
    Vol(const std::string &filename);
    ~Vol() {}
    Eigen::Vector3i dims;
    Eigen::Vector3f voxel_spacing;
    Eigen::Vector3f origin;
    Eigen::Vector3f s;
    Eigen::Vector3f vs_inv;
    Eigen::Affine3f transformation;
    int *data;
    int operator()(int i, int j, int k) const {
      return data[i + dims[0] * j + dims[0] * dims[1] * k];
    }
  };
  Vol vol;
};
