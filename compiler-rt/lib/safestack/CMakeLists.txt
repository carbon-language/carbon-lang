add_custom_target(safestack)

set(SAFESTACK_SOURCES safestack.cc)

include_directories(..)

set(SAFESTACK_CFLAGS ${SANITIZER_COMMON_CFLAGS})

if(APPLE)
  # Build universal binary on APPLE.
  add_compiler_rt_osx_static_runtime(clang_rt.safestack_osx
    ARCH ${SAFESTACK_SUPPORTED_ARCH}
    SOURCES ${SAFESTACK_SOURCES}
            $<TARGET_OBJECTS:RTInterception.osx>
            $<TARGET_OBJECTS:RTSanitizerCommon.osx>
    CFLAGS ${SAFESTACK_CFLAGS})
  add_dependencies(safestack clang_rt.safestack_osx)
else()
  # Otherwise, build separate libraries for each target.
  foreach(arch ${SAFESTACK_SUPPORTED_ARCH})
    add_compiler_rt_runtime(clang_rt.safestack-${arch} ${arch} STATIC
      SOURCES ${SAFESTACK_SOURCES}
              $<TARGET_OBJECTS:RTInterception.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
      CFLAGS ${SAFESTACK_CFLAGS})
    add_dependencies(safestack clang_rt.safestack-${arch})
  endforeach()
endif()
