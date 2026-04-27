find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(smr smr.cc ../include/kernels/matrix_ops.cu)

target_link_libraries(smr
  PRIVATE
    ${PROJECT_NAME} 
    ${CMAKE_SOURCE_DIR}/lib/libcrashconsensus.so
    CUDA::cudart
    OpenMP::OpenMP_CXX
    ibverbs
)

message(STATUS "Building with Mu enabled...")
set_target_properties(smr PROPERTIES
  BUILD_RPATH "${CMAKE_SOURCE_DIR}/lib"
  INSTALL_RPATH "${CMAKE_SOURCE_DIR}/lib"
)