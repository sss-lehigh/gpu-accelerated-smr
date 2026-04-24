add_executable(smr smr.cc)

target_link_libraries(smr
  PRIVATE
    ${PROJECT_NAME} 
    ${CMAKE_SOURCE_DIR}/lib/libcrashconsensus.so
)

message(STATUS "Building with Mu enabled...")
set_target_properties(smr PROPERTIES
  BUILD_RPATH "${CMAKE_SOURCE_DIR}/lib"
  INSTALL_RPATH "${CMAKE_SOURCE_DIR}/lib"
)