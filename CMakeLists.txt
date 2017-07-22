cmake_minimum_required(VERSION 2.8)
 
PROJECT(GPUstreamline)

find_package(CUDA REQUIRED)
CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})


cuda_add_executable(GPUstreamline GPUstreamline.cu)
#add_executable(GPUstreamline MACOSX_BUNDLE GPUstreamline.cu)

if(VTK_LIBRARIES)
  target_link_libraries(GPUstreamline ${VTK_LIBRARIES})
else()
  target_link_libraries(GPUstreamline vtkHybrid vtkWidgets)
endif()
