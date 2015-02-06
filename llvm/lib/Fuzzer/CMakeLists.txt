# Disable the coverage instrumentation for the fuzzer itself.
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -fsanitize-coverage=0")
if( LLVM_USE_SANITIZE_COVERAGE  )
  add_library(LLVMFuzzerNoMain OBJECT
    FuzzerCrossOver.cpp
    FuzzerIO.cpp
    FuzzerLoop.cpp
    FuzzerMutate.cpp
    FuzzerSanitizerOptions.cpp
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
