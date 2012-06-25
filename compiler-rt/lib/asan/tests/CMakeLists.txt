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

# FIXME: Currently, this detection isn't working. Assume we're doing
# a bootstrap build for now.
set(HOST_HAS_ASAN on)
#check_cxx_compiler_flag("-faddress-sanitizer" HOST_HAS_ASAN)

if(HOST_HAS_ASAN)
	add_asan_test(AsanInstrumentationTests
		asan_globals_test.cc
		asan_test.cc
		asan_break_optimization.cc
		)
	set_property(TARGET AsanInstrumentationTests APPEND_STRING PROPERTY COMPILE_FLAGS
		" -faddress-sanitizer ${ASAN_CFLAGS}")
	set_property(TARGET AsanInstrumentationTests APPEND_STRING PROPERTY COMPILE_FLAGS
		" -mllvm -asan-blacklist=${CMAKE_CURRENT_SOURCE_DIR}/asan_test.ignore")
	set_property(TARGET AsanInstrumentationTests APPEND PROPERTY COMPILE_DEFINITIONS
		ASAN_HAS_BLACKLIST=1
		ASAN_HAS_EXCEPTIONS=1
		ASAN_NEEDS_SEGV=1
		ASAN_UAR=0
		)
endif()

