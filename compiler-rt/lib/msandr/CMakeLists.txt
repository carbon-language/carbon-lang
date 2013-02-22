
if(DynamoRIO_DIR AND DrMemoryFramework_DIR)
  set(CMAKE_COMPILER_IS_GNUCC 1)
  find_package(DynamoRIO)
  find_package(DrMemoryFramework)

  set(arch "x86_64")
  add_library(clang_rt.msandr-${arch} SHARED msandr.cc)
  configure_DynamoRIO_client(clang_rt.msandr-${arch})

  function(append_target_cflags tgt cflags)
    get_property(old_cflags TARGET clang_rt.msandr-${arch} PROPERTY COMPILE_FLAGS)
    set_property(TARGET clang_rt.msandr-${arch} PROPERTY COMPILE_FLAGS "${old_cflags} ${cflags}")
  endfunction(append_target_cflags)

  append_target_cflags(clang_rt.msandr-${arch} "-Wno-c++11-extensions")

  use_DynamoRIO_extension(clang_rt.msandr-${arch} drutil)
  use_DynamoRIO_extension(clang_rt.msandr-${arch} drmgr)
  use_DynamoRIO_extension(clang_rt.msandr-${arch} drsyscall)

  set_target_properties(clang_rt.msandr-${arch} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${COMPILER_RT_LIBRARY_OUTPUT_DIR})
  install(TARGETS clang_rt.msandr-${arch}
    LIBRARY DESTINATION ${COMPILER_RT_LIBRARY_INSTALL_DIR})
endif()
