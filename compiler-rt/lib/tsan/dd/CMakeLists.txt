# Build for the experimental deadlock detector runtime library.

include_directories(../..)

set(DD_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_no_rtti_flag(DD_CFLAGS)

if("${CMAKE_BUILD_TYPE}" EQUAL "Release")
  set(DD_COMMON_DEFINITIONS DEBUG=0)
else()
  set(DD_COMMON_DEFINITIONS DEBUG=1)
endif()

set(DD_DYNAMIC_DEFINITIONS DYNAMIC=1)

set(DD_SOURCES
  dd_rtl.cc
  dd_interceptors.cc
)

set(DD_HEADERS
  dd_rtl.h
)

add_custom_target(dd)
# Deadlock detector is currently supported on 64-bit Linux only.
if(CAN_TARGET_x86_64 AND UNIX AND NOT APPLE AND NOT ANDROID)
  set(arch "x86_64")
  add_compiler_rt_static_runtime(clang_rt.dd-${arch} ${arch}
    SOURCES ${DD_SOURCES}
            $<TARGET_OBJECTS:RTInterception.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
    CFLAGS ${DD_CFLAGS}
    DEFS ${DD_COMMON_DEFINITIONS})

  add_library(RTDD OBJECT ${DD_SOURCES})
  set_target_compile_flags(RTDD ${DD_CFLAGS})
  set_property(TARGET RTDD APPEND PROPERTY
    COMPILE_DEFINITIONS ${DD_COMMON_DEFINITIONS})
  set_property(TARGET RTDD APPEND PROPERTY
    COMPILE_DEFINITIONS ${DD_DYNAMIC_DEFINITIONS})

  add_library(clang_rt.dyndd-${arch} SHARED
    $<TARGET_OBJECTS:RTDD>
    $<TARGET_OBJECTS:RTInterception.${arch}>
    $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
    $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>)
#  set_target_compile_flags(clang_rt.dd-${arch} ${DD_CFLAGS})
#  set_property(TARGET clang_rt.dd-${arch} APPEND PROPERTY
#    COMPILE_DEFINITIONS ${DD_COMMON_DEFINITIONS})
  target_link_libraries(clang_rt.dyndd-${arch} pthread dl)
endif()

add_dependencies(compiler-rt dd)

