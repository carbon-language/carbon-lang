# These tests are not instrumented with coverage.

set(CMAKE_CXX_FLAGS
  "${LIBFUZZER_FLAGS_BASE} -fno-sanitize=all -fno-sanitize-coverage=edge,trace-cmp,indirect-calls,8bit-counters")

foreach(Test ${UninstrumentedTests})
  add_executable(LLVMFuzzer-${Test}-Uninstrumented
    ../${Test}.cpp
    )
  target_link_libraries(LLVMFuzzer-${Test}-Uninstrumented
    LLVMFuzzer
    )
endforeach()

