if(MSVC)
  # For tests use the CRT specified for release build
  # (asan doesn't support MDd and MTd)
  if ("${LLVM_USE_CRT_RELEASE}" STREQUAL "")
    set(CRT_FLAG " /MD ")
  else()
    set(CRT_FLAG " /${LLVM_USE_CRT_RELEASE} ")
  endif()
  # In order to use the sanitizers in Windows, we need to link against many
  # runtime libraries which will depend on the target being created
  # (executable or dll) and the c runtime library used (MT/MD).
  # By default, cmake uses link.exe for linking, which fails because we don't
  # specify the appropiate dependencies.
  # As we don't want to consider all of that possible situations which depends
  # on the implementation of the compiler-rt, the simplest option is to change
  # the rules for linking executables and shared libraries, using the compiler
  # instead of link.exe. Clang will consider the sanitizer flags, and
  # automatically provide the required libraries to the linker.
  set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_CXX_COMPILER> <FLAGS> ${CMAKE_CXX_FLAGS} ${CRT_FLAG} <OBJECTS> -o <TARGET> <LINK_LIBRARIES> /link <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS>")
  set(CMAKE_CXX_CREATE_SHARED_LIBRARY "<CMAKE_CXX_COMPILER> ${CMAKE_CXX_FLAGS} ${CRT_FLAG} /LD <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG> <TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES> /link <LINK_FLAGS>")
endif()

if(APPLE OR MSVC)
  # LeakSanitizer is not supported on OSX and Windows right now
  set(HAS_LSAN 0)
  message(WARNING "LeakSanitizer is not supported."
    " Building and running LibFuzzer LeakSanitizer tests is disabled."
    )
else()
  set(HAS_LSAN 1)
endif()

###############################################################################
# Unit tests
###############################################################################

add_custom_target(FuzzerUnitTests)
set_target_properties(FuzzerUnitTests PROPERTIES FOLDER "libFuzzer tests")

add_executable(LLVMFuzzer-Unittest FuzzerUnittest.cpp)

target_link_libraries(LLVMFuzzer-Unittest
  gtest
  gtest_main
  LLVMFuzzerNoMain
  )

target_include_directories(LLVMFuzzer-Unittest PRIVATE
  "${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include"
  )

set_target_properties(LLVMFuzzer-Unittest
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY
  "${CMAKE_CURRENT_BINARY_DIR}"
)

include_directories(..)

###############################################################################
# Configure lit to run the tests
#
# Note this is done after declaring all tests so we can inform lit if any tests
# need to be disabled.
###############################################################################
set(LIBFUZZER_POSIX 1)
if (MSVC)
  set(LIBFUZZER_POSIX 0)
endif()

# Use just-built Clang to compile/link tests on all platforms, except for
# Windows where we need to use clang-cl instead.
if(NOT MSVC)
  set(LIBFUZZER_TEST_COMPILER ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang)
  set(LIBFUZZER_TEST_CXX_COMPILER ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang++)
else()
  set(LIBFUZZER_TEST_COMPILER ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang.exe)
  set(LIBFUZZER_TEST_CXX_COMPILER ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang++.exe)
endif()

# LIT-based libFuzzer tests.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

# libFuzzer unit tests.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/unit/lit.site.cfg
  )

add_lit_testsuite(check-fuzzer "Running Fuzzer tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS LLVMFuzzer-Unittest
    )


# Don't add dependencies on Windows. The linker step would fail on Windows,
# since cmake will use link.exe for linking and won't include compiler-rt libs.
if(NOT MSVC)
  add_dependencies(check-fuzzer LLVMFuzzer asan clang llvm-symbolizer FileCheck sancov not)
endif()
