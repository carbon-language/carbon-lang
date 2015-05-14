set(LIBFUZZER_FLAGS_BASE "${CMAKE_CXX_FLAGS_RELEASE}")
# Disable the coverage and sanitizer instrumentation for the fuzzer itself.
set(CMAKE_CXX_FLAGS_RELEASE "${LIBFUZZER_FLAGS_BASE} -O2 -fno-sanitize=all")
if( LLVM_USE_SANITIZE_COVERAGE )
  add_library(LLVMFuzzerNoMain OBJECT
    FuzzerCrossOver.cpp
    FuzzerTraceState.cpp
    FuzzerDriver.cpp
    FuzzerIO.cpp
    FuzzerLoop.cpp
    FuzzerMutate.cpp
    FuzzerSanitizerOptions.cpp
    FuzzerSHA1.cpp
    FuzzerUtil.cpp
    )
  add_library(LLVMFuzzer STATIC
    FuzzerMain.cpp
    $<TARGET_OBJECTS:LLVMFuzzerNoMain>
    )

  if( LLVM_INCLUDE_TESTS )
    add_subdirectory(test)
  endif()
endif()
