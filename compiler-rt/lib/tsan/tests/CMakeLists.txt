include_directories(../rtl)

add_custom_target(TsanUnitTests)
set_target_properties(TsanUnitTests PROPERTIES
  FOLDER "TSan unittests")

set(TSAN_UNITTEST_CFLAGS
  ${TSAN_CFLAGS}
  ${COMPILER_RT_GTEST_INCLUDE_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/lib
  -I${COMPILER_RT_SOURCE_DIR}/lib/tsan/rtl
  -DGTEST_HAS_RTTI=0)

# tsan_compile(obj_list, source, arch, {headers})
macro(tsan_compile obj_list source arch)
  get_filename_component(basename ${source} NAME)
  set(output_obj "${basename}.${arch}.o")
  get_target_flags_for_arch(${arch} TARGET_CFLAGS)
  clang_compile(${output_obj} ${source}
          CFLAGS ${TSAN_UNITTEST_CFLAGS} ${TARGET_CFLAGS}
          DEPS gtest tsan ${ARGN})
  list(APPEND ${obj_list} ${output_obj})
endmacro()

macro(add_tsan_unittest testname)
  # Build unit tests only for 64-bit Linux.
  if(UNIX AND NOT APPLE AND CAN_TARGET_x86_64)
    parse_arguments(TEST "SOURCES;HEADERS" "" ${ARGN})
    set(TEST_OBJECTS)
    foreach(SOURCE ${TEST_SOURCES} ${COMPILER_RT_GTEST_SOURCE})
      tsan_compile(TEST_OBJECTS ${SOURCE} x86_64 ${TEST_HEADERS})
    endforeach()
    get_target_flags_for_arch(${arch} TARGET_LINK_FLAGS)
    add_compiler_rt_test(TsanUnitTests ${testname}
            OBJECTS ${TEST_OBJECTS}
            DEPS tsan ${TEST_OBJECTS}
            LINK_FLAGS ${TARGET_LINK_FLAGS}
                       -fsanitize=thread
                       -lstdc++ -lm)
  endif()
endmacro()

if(COMPILER_RT_CAN_EXECUTE_TESTS)
  add_subdirectory(rtl)
  add_subdirectory(unit)
endif()
