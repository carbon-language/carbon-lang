include(ExternalProject)

set(BOLT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BOLT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_CXX_STANDARD 14)

ExternalProject_Add(bolt_rt
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/runtime"
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/bolt_rt-stamps
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/bolt_rt-bins
  CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_BUILD_TYPE=Release
             -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
             -DCMAKE_INSTALL_PREFIX=${LLVM_BINARY_DIR}
  # You might want to set this to True if actively developing bolt_rt, otherwise
  # cmake will not rebuild it after source code changes
  BUILD_ALWAYS True
  )

install(CODE "execute_process\(COMMAND \${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=\${CMAKE_INSTALL_PREFIX} -P ${CMAKE_CURRENT_BINARY_DIR}/bolt_rt-bins/cmake_install.cmake \)"
  COMPONENT bolt_rt)

add_llvm_install_targets(install-bolt_rt
  DEPENDS bolt_rt
  COMPONENT bolt_rt)

add_subdirectory(src)
add_subdirectory(test)
