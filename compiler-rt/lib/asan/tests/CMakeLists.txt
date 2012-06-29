# Testing rules for AddressSanitizer.
#
# These are broken into two buckets. One set of tests directly interacts with
# the runtime library and checks its functionality. These are the
# no-instrumentation tests.
#
# Another group of tests relies upon the ability to compile the test with
# address sanitizer instrumentation pass. These tests form "integration" tests
# and have some elements of version skew -- they test the *host* compiler's
# instrumentation against the just-built runtime library.

include(CheckCXXCompilerFlag)

include_directories(..)
include_directories(../..)

set(ASAN_UNITTEST_COMMON_CFLAGS
  -Wall
  -Wno-format
  -fvisibility=hidden
)

set(ASAN_GTEST_INCLUDE_CFLAGS
  -I${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include
  -I${LLVM_MAIN_SRC_DIR}/include
  -I${LLVM_BINARY_DIR}/include
  -D__STDC_CONSTANT_MACROS
  -D__STDC_LIMIT_MACROS
)

set(ASAN_UNITTEST_INSTRUMENTED_CFLAGS
  ${ASAN_UNITTEST_COMMON_CFLAGS}
  ${ASAN_GTEST_INCLUDE_CFLAGS}
  -faddress-sanitizer
  -O2
  -g
  -mllvm "-asan-blacklist=${CMAKE_CURRENT_SOURCE_DIR}/asan_test.ignore"
  -DASAN_HAS_BLACKLIST=1
  -DASAN_HAS_EXCEPTIONS=1
  -DASAN_NEEDS_SEGV=1
  -DASAN_UAR=0
)

add_custom_target(AsanTests)
set_target_properties(AsanTests PROPERTIES FOLDER "ASan tests")
function(add_asan_test testname)
	add_unittest(AsanTests ${testname} ${ARGN})
	if(CMAKE_SIZEOF_VOID_P EQUAL 4)
		target_link_libraries(${testname} clang_rt.asan-i386)
	else()
		target_link_libraries(${testname} clang_rt.asan-x86_64)
	endif()
	if (APPLE)
		# Darwin-specific linker flags.
		set_property(TARGET ${testname} APPEND PROPERTY
		  LINK_FLAGS "-framework CoreFoundation")
	elseif (UNIX)
		# Linux-specific linker flags.
		set_property(TARGET ${testname} APPEND PROPERTY
		  LINK_FLAGS "-lpthread -ldl -export-dynamic")
	endif()
	get_property(compile_flags TARGET ${testname} PROPERTY COMPILE_FLAGS)
	foreach(arg ${ASAN_UNITTEST_COMMON_CFLAGS})
	  set(add_compile_flags "${add_compile_flags} ${arg}")
	endforeach(arg ${ASAN_UNITTEST_COMMON_CFLAGS})
	set_property(TARGET ${testname} PROPERTY COMPILE_FLAGS
	  "${compile_flags} ${add_compile_flags}")
endfunction()

set(ASAN_TEST_FILES
  asan_noinst_test.cc
  asan_break_optimization.cc
)

# We only support building instrumented tests when we're not cross compiling
# and targeting a unix-like system where we can predict viable compilation and
# linking strategies.
if("${CMAKE_HOST_SYSTEM}" STREQUAL "${CMAKE_SYSTEM}" AND UNIX)

  # This function is a custom routine to manage manually compiling source files
  # for unit tests with the just-built Clang binary, using the ASan
  # instrumentation, and linking them into a test executable.
  function(add_asan_compile_cxx_command source)
    add_custom_command(
      OUTPUT "${source}.asan.o"
      COMMAND clang
              ${ASAN_UNITTEST_INSTRUMENTED_CFLAGS}
              -c -o "${source}.asan.o"
              ${CMAKE_CURRENT_SOURCE_DIR}/${source}
      MAIN_DEPENDENCY ${source}
      DEPENDS clang ${ARGN}
      )
  endfunction()

  function(add_asan_compile_objc_command source)
    add_custom_command(
      OUTPUT "${source}.asan.o"
      COMMAND clang
              ${ASAN_UNITTEST_INSTRUMENTED_CFLAGS}
              -ObjC -c -o "${source}.asan.o"
              ${CMAKE_CURRENT_SOURCE_DIR}/${source}
      MAIN_DEPENDENCY ${source}
      DEPENDS clang ${ARGN}
      )
  endfunction()

  add_asan_compile_cxx_command(asan_globals_test.cc)
  add_asan_compile_cxx_command(asan_test.cc)
  list(APPEND ASAN_TEST_FILES
    asan_globals_test.cc.asan.o
    asan_test.cc.asan.o
  )
  if (APPLE)
    add_asan_compile_objc_command(asan_mac_test.mm)
    list(APPEND ASAN_TEST_FILES asan_mac_test.mm.asan.o)
  endif()

endif()

add_asan_test(AsanTest
  ${ASAN_TEST_FILES}
)
