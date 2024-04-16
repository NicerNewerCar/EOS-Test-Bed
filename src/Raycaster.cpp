#include "Raycaster.hpp"
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

  transformation = Eigen::Affine3f::Identity();
}

void RayCaster::render(float *buffer, int C, int R, int z0) {
  Eigen::Vector3f cameraPos(-sid, 0, 0);
  if (is_lateral) {
    cameraPos[0] = 0.f;
    cameraPos[1] = -sid;
  }

  Eigen::Vector3f s_transformed = (vol.transformation * vol.s).cwiseAbs();

#pragma omp parallel for
  for (int v = 0; v < R; ++v) {
    for (int u = 0; u < C; ++u) {
      cameraPos[2] = z0 - v * lamdba_vert;

      // Frontal geometry
      Eigen::Vector3f detector_position(
          sdd - sid, ((u - (C / 2.f)) * lamdba_hori * sdd) / sid, cameraPos[2]);
      if (is_lateral) {
        detector_position[0] = ((u - (C / 2.f)) * lamdba_hori * sdd) / sid;
        detector_position[1] = sdd - sid;
      }

      Eigen::Vector3f ray_direction = detector_position - cameraPos;
      ray_direction.normalize();

      float near;
      float far;
      rayBoxIntersection(cameraPos, ray_direction, s_transformed, near, far);

      if (near > far) { // Missed
        buffer[C * v + u] = 0.f;
        continue;
      }

      Eigen::Affine3f vol_transformed =
          vol.transformation.inverse() * Eigen::Translation3f(cameraPos);

      // Ray Marching
      Eigen::Vector3f p;
      float step = 0.1;
      float t = near;
      float sum = 0;
      while (t < far) {
        // p = cameraPos + t * ray_direction;
        p = vol_transformed * (t * ray_direction);
        t += step;

        // Check to see if we are out of the CT
        if (p.x() <= -s_transformed.x() || p.x() >= s_transformed.x() - 2)
          continue;
        if (p.y() <= -s_transformed.y() || p.y() >= s_transformed.y() - 2)
          continue;
        if (p.z() <= -s_transformed.z() || p.z() >= s_transformed.z() - 2)
          continue;

        // If inside the CT, we convert our coordinates to the CT coordinates
        Eigen::Vector3f p_model((p.x() + s_transformed.x()) * vol.vs_inv.x(),
                                (p.y() + s_transformed.y()) * vol.vs_inv.y(),
                                (p.z() + s_transformed.z()) * vol.vs_inv.z());

        sum += trilinear_interpolation(p_model);
      }
      double val = sum / 10.f;
      buffer[C * v + u] = val;
    }
  }
}

float RayCaster::trilinear_interpolation(Eigen::Vector3f p) {
  // See https://en.wikipedia.org/wiki/Trilinear_interpolation
  Eigen::Vector3i p000 = p.cast<int>();
  Eigen::Vector3i p111 = p000 + Eigen::Vector3i::Ones();
  Eigen::Vector3i p001(p000[0], p000[1], p111[2]);
  Eigen::Vector3i p010(p000[0], p111[1], p000[2]);
  Eigen::Vector3i p100(p111[0], p000[1], p000[2]);
  Eigen::Vector3i p011(p000[0], p111[1], p111[2]);
  Eigen::Vector3i p101(p111[0], p000[1], p111[2]);
  Eigen::Vector3i p110(p111[0], p111[1], p000[2]);

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
                                   Eigen::Vector3f ray_direction,
                                   Eigen::Vector3f s_transformed, float &near,
                                   float &far) {
  // See https://education.siggraph.org/static/HyperGraph/raytrace/rtinter3.htm
  Eigen::Vector3f boxMin(-s_transformed.x(), -s_transformed.y(),
                         -s_transformed.z());
  Eigen::Vector3f boxMax = s_transformed.array();

  Eigen::Vector3f invRayDir = ray_direction.array().inverse();
  Eigen::Vector3f t1 = (boxMin - ray_origin).array() * invRayDir.array();
  Eigen::Vector3f t2 = (boxMax - ray_origin).array() * invRayDir.array();

  Eigen::Vector3f tMin = t1.array().min(t2.array());
  Eigen::Vector3f tMax = t1.array().max(t2.array());

  near = tMin.maxCoeff();
  far = tMax.minCoeff();
}
