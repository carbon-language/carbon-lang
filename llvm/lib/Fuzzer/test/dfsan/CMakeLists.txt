# These tests depend on both coverage and dfsan instrumentation.

set(CMAKE_CXX_FLAGS
  "${LIBFUZZER_FLAGS_BASE} -fno-sanitize=all -fsanitize=dataflow")

foreach(Test ${DFSanTests})
  add_executable(LLVMFuzzer-${Test}-DFSan
    ../${Test}.cpp
    )
  target_link_libraries(LLVMFuzzer-${Test}-DFSan
    LLVMFuzzer
    )
endforeach()

