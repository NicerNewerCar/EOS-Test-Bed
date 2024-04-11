#include "Raycaster.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <teem/nrrd.h>

RayCaster::Vol::Vol(const std::string &filename) {
  Nrrd *nrrd = nrrdNew();
  nrrdLoad(nrrd, filename.c_str(), NULL);

  this->voxel_spacing[0] = nrrd->axis[0].spaceDirection[0];
  this->voxel_spacing[1] = nrrd->axis[1].spaceDirection[1];
  this->voxel_spacing[2] = nrrd->axis[2].spaceDirection[2];

  this->dims[0] = nrrd->axis[0].size;
  this->dims[1] = nrrd->axis[1].size;
  this->dims[2] = nrrd->axis[2].size;

  this->origin[0] = nrrd->spaceOrigin[0];
  this->origin[1] = nrrd->spaceOrigin[1];
  this->origin[2] = nrrd->spaceOrigin[2];

  this->s[0] = this->dims[0] * this->voxel_spacing[0] / 2;
  this->s[1] = this->dims[1] * this->voxel_spacing[1] / 2;
  this->s[2] = this->dims[2] * this->voxel_spacing[2] / 2;

  this->vs_inv[0] = 1.f / this->voxel_spacing[0];
  this->vs_inv[1] = 1.f / this->voxel_spacing[1];
  this->vs_inv[2] = 1.f / this->voxel_spacing[2];

  size_t data_size =
      this->dims[0] * this->dims[1] * this->dims[2] * sizeof(int);
  this->data = new int[data_size];
  memcpy((void *)this->data, nrrd->data, data_size);
}

void RayCaster::render(float *buffer, int C, int R, int z0) {
  Eigen::Vector3f cameraPos(-sid, 0, 0);
  if (is_lateral) {
    cameraPos[0] = 0.f;
    cameraPos[1] = -sid;
  }
#pragma omp parallel for
  for (int v = 0; v < R; ++v) {
    for (int u = 0; u < C; ++u) {
      cameraPos[2] = lamdba_vert * v - z0;

      // Frontal geometry
      Eigen::Vector3f ray_direction(
          sdd - sid, ((u - (C / 2.f)) * lamdba_hori * sdd) / sid, cameraPos[2]);
      if (is_lateral) {
        ray_direction[0] = ((u - (C / 2.f)) * lamdba_hori * sdd) / sid;
        ray_direction[1] = sdd - sid;
      }

      float near;
      float far;
      rayBoxIntersection(cameraPos, ray_direction, near, far);

      if (near > far) { // Missed
        buffer[R * u + v] = 0.f;
        continue;
      }

      // Ray Marching
      Eigen::Vector3f p;
      float step = 0.001;
      float t = near;
      float sum = 0;
      while (t < far) {
        p = cameraPos + t * ray_direction;
        t += step;

        // Check to see if we are out of the CT
        if (p.x() <= -vol.s.x() || p.x() >= vol.s.x() - 2)
          continue;
        if (p.y() <= -vol.s.y() || p.y() >= vol.s.y() - 2)
          continue;
        if (p.z() <= -vol.s.z() || p.z() >= vol.s.z() - 2)
          continue;

        // If inside the CT, we convert our coordinates to the CT coordinates
        Eigen::Vector3f p_model((p.x() + vol.s.x()) * vol.vs_inv.x(),
                                (p.y() + vol.s.y()) * vol.vs_inv.y(),
                                (p.z() + vol.s.z()) * vol.vs_inv.z());

        sum += trilinear_interpolation(p_model);
      }
      double val = sum / 10.f;
      // buffer[R * u + v] = val;
      buffer[C * v + u] = val;
    }
  }
}

float RayCaster::trilinear_interpolation(Eigen::Vector3f p) {
  // See https://en.wikipedia.org/wiki/Trilinear_interpolation
  Eigen::Vector3i p000 = Eigen::Vector3i(floor(p[0]), floor(p[1]), floor(p[2]));
  Eigen::Vector3i p111 = Eigen::Vector3i(p000[0] + 1, p000[1] + 1, p000[2] + 1);
  Eigen::Vector3i p011 = Eigen::Vector3i(p000[0], p000[1] + 1, p000[2] + 1);
  Eigen::Vector3i p001 = Eigen::Vector3i(p000[0], p000[1], p000[2] + 1);
  Eigen::Vector3i p101 = Eigen::Vector3i(p000[0] + 1, p000[1], p000[2] + 1);
  Eigen::Vector3i p100 = Eigen::Vector3i(p000[0] + 1, p000[1], p000[2]);
  Eigen::Vector3i p110 = Eigen::Vector3i(p000[0] + 1, p000[1] + 1, p000[2]);
  Eigen::Vector3i p010 = Eigen::Vector3i(p000[0], p000[1] + 1, p000[2]);

  float u000 = vol(p000[0], p000[1], p000[2]);
  float u100 = vol(p100[0], p100[1], p100[2]);
  float u010 = vol(p010[0], p010[1], p010[2]);
  float u101 = vol(p101[0], p101[1], p101[2]);
  float u001 = vol(p001[0], p001[1], p001[2]);
  float u110 = vol(p110[0], p110[1], p110[2]);
  float u011 = vol(p011[0], p011[1], p011[2]);
  float u111 = vol(p111[0], p111[1], p111[2]);

  float xd = p[0] - p000[0];
  float yd = p[1] - p000[1];
  float zd = p[2] - p000[2];

  float c00 = u000 * (1 - xd) + u100 * xd;
  float c01 = u001 * (1 - xd) + u101 * xd;
  float c10 = u010 * (1 - xd) + u110 * xd;
  float c11 = u011 * (1 - xd) + u111 * xd;

  float c0 = c00 * (1 - yd) + c10 * yd;
  float c1 = c01 * (1 - yd) + c11 * yd;

  float c = c0 * (1 - zd) + c1 * zd;

  return c;
}

void RayCaster::rayBoxIntersection(Eigen::Vector3f ray_origin,
                                   Eigen::Vector3f ray_direction, float &near,
                                   float &far) {
  // See https://education.siggraph.org/static/HyperGraph/raytrace/rtinter3.htm
  Eigen::Vector3f boxMin(-vol.s.x(), -vol.s.y(), -vol.s.z());
  Eigen::Vector3f boxMax(vol.s.x(), vol.s.y(), vol.s.z());

  Eigen::Vector3f tBot((boxMin.x() - ray_origin.x()) / ray_direction.x(),
                       (boxMin.y() - ray_origin.y()) / ray_direction.y(),
                       (boxMin.z() - ray_origin.z()) / ray_direction.z());

  Eigen::Vector3f tTop((boxMax.x() - ray_origin.x()) / ray_direction.x(),
                       (boxMax.y() - ray_origin.y()) / ray_direction.y(),
                       (boxMax.z() - ray_origin.z()) / ray_direction.z());

  Eigen::Vector3f tMin(std::min(tBot.x(), tTop.x()),
                       std::min(tBot.y(), tTop.y()),
                       std::min(tBot.z(), tTop.z()));

  Eigen::Vector3f tMax(std::max(tBot.x(), tTop.x()),
                       std::max(tBot.y(), tTop.y()),
                       std::max(tBot.z(), tTop.z()));

  near = tMin.maxCoeff();
  far = tMax.minCoeff();
}
