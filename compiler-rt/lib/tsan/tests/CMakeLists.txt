include_directories(../rtl)

add_custom_target(TsanUnitTests)
set_target_properties(TsanUnitTests PROPERTIES
  FOLDER "TSan unittests")

set(TSAN_UNITTEST_CFLAGS
  ${TSAN_CFLAGS}
  ${COMPILER_RT_TEST_CFLAGS}
  ${COMPILER_RT_GTEST_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/lib
  -I${COMPILER_RT_SOURCE_DIR}/lib/tsan/rtl
  -DGTEST_HAS_RTTI=0)

set(TSAN_RTL_HEADERS)
foreach (header ${TSAN_HEADERS})
  list(APPEND TSAN_RTL_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/../${header})
endforeach()

# tsan_compile(obj_list, source, arch, {headers})
macro(tsan_compile obj_list source arch)
  get_filename_component(basename ${source} NAME)
  set(output_obj "${basename}.${arch}.o")
  get_target_flags_for_arch(${arch} TARGET_CFLAGS)
  set(COMPILE_DEPS ${TSAN_RTL_HEADERS} ${ARGN})
  if(NOT COMPILER_RT_STANDALONE_BUILD)
    list(APPEND COMPILE_DEPS gtest tsan)
  endif()
  clang_compile(${output_obj} ${source}
          CFLAGS ${TSAN_UNITTEST_CFLAGS} ${TARGET_CFLAGS}
          DEPS ${COMPILE_DEPS})
  list(APPEND ${obj_list} ${output_obj})
endmacro()

macro(add_tsan_unittest testname)
  # Build unit tests only for 64-bit Linux.
  if(UNIX AND NOT APPLE)
    foreach(arch ${TSAN_SUPPORTED_ARCH})
      cmake_parse_arguments(TEST "" "" "SOURCES;HEADERS" ${ARGN})
      set(TEST_OBJECTS)
      foreach(SOURCE ${TEST_SOURCES} ${COMPILER_RT_GTEST_SOURCE})
        tsan_compile(TEST_OBJECTS ${SOURCE} ${arch} ${TEST_HEADERS})
      endforeach()
      get_target_flags_for_arch(${arch} TARGET_LINK_FLAGS)
      set(TEST_DEPS ${TEST_OBJECTS})
      if(NOT COMPILER_RT_STANDALONE_BUILD)
        list(APPEND TEST_DEPS tsan)
      endif()
      # FIXME: Looks like we should link TSan with just-built runtime,
      # and not rely on -fsanitize=thread, as these tests are essentially
      # unit tests.
      add_compiler_rt_test(TsanUnitTests ${testname}
              OBJECTS ${TEST_OBJECTS}
              DEPS ${TEST_DEPS}
              LINK_FLAGS ${TARGET_LINK_FLAGS}
                         -fsanitize=thread
                         -lstdc++ -lm)
    endforeach()
  endif()
endmacro()

if(COMPILER_RT_CAN_EXECUTE_TESTS)
  add_subdirectory(rtl)
  add_subdirectory(unit)
endif()
