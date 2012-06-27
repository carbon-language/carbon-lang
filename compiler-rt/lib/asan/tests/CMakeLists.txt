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

add_custom_target(AsanTests)
set_target_properties(AsanTests PROPERTIES FOLDER "ASan tests")
function(add_asan_test testname)
	add_unittest(AsanTests ${testname} ${ARGN})
	if(CMAKE_SIZEOF_VOID_P EQUAL 4)
		target_link_libraries(${testname} clang_rt.asan-i386)
	else()
		target_link_libraries(${testname} clang_rt.asan-x86_64)
	endif()
endfunction()

add_asan_test(AsanNoInstrumentationTests
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
  function(add_asan_compile_command source)
    add_custom_command(
      OUTPUT "${source}.asan.o"
      COMMAND clang
              -faddress-sanitizer ${ASAN_CFLAGS}
              -mllvm "-asan-blacklist=${CMAKE_CURRENT_SOURCE_DIR}/asan_test.ignore"
              -DASAN_HAS_BLACKLIST=1
              -DASAN_HAS_EXCEPTIONS=1
              -DASAN_NEEDS_SEGV=1
              -DASAN_UAR=0
              -c -o "${source}.asan.o"
              ${CMAKE_CURRENT_SOURCE_DIR}/${source}
      MAIN_DEPENDENCY ${source}
      DEPENDS clang ${ARGN}
      )
  endfunction()

  add_asan_compile_command(asan_globals_test.cc)
  add_asan_compile_command(asan_test.cc)

	add_asan_test(AsanInstrumentationTests
		asan_globals_test.cc.asan.o
		asan_test.cc.asan.o
		asan_break_optimization.cc
		)

endif()
