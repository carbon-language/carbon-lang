# These tests are not instrumented with coverage.

set(CMAKE_CXX_FLAGS
  "${LIBFUZZER_FLAGS_BASE} -fno-sanitize-coverage=8bit-counters -fsanitize-coverage=trace-pc")

foreach(Test ${TracePCTests})
  add_executable(LLVMFuzzer-${Test}-TracePC
    ../${Test}.cpp
    )
  target_link_libraries(LLVMFuzzer-${Test}-TracePC
    LLVMFuzzer
    )
endforeach()

