# EOS Test Bed
This repo servers as a small test library to test out the EOS projection model.

## Budilding

Requires:
* CMake 3.8 or higher
* C++ 17 or higher
* OpenCL 1.2 or higher

Note: This requires OpenCL ICD Loader, OpenCL Headers, and LIBTIFF. I reference the versions of this libraries that build with Autoscoper's superbuild.

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=<same type that Autoscoper was built with> \
     -DOpenCLHeaders_DIR=<autoscoper-build>\OpenCL-Headers-build\OpenCLHeaders \
     -DOpenCLICDLoader_DIR=<autoscoper-build>OpenCL-ICD-Loader-build\OpenCLICDLoader \
     -DTIFF_INCLUDE_DIR=<autoscoper-build>/TIFF-install/include \
     -DTIFF_LIBRARY:UNINITIALIZED=<autoscoper-build>TIFF-install\lib\tiff.lib \
```
Make sure the `TIFF_LIBRARY` is referencing either the debug or release version of the library depending on the build type.


## Running
```
./EOS <volume name> <output name (.pgm)> <is_lateral> <vol_dim_x> <vol_dim_y> <vol_dim_z>
```

* `volume name` - The path to the volume to project (tif stack)
* `output name` - The path to the output file (this will be a pgm file)
* `is_lateral` - 0 to project the volume in the frontal direction, 1 to project the volume in the lateral direction
* `vol_dim_x` - The x dimension of the volume
* `vol_dim_y` - The y dimension of the volume
* `vol_dim_z` - The z dimension of the volume