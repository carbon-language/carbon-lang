# These tests depend on both coverage and dfsan instrumentation.

set(DFSAN_FUZZER_ABI_LIST "${CMAKE_CURRENT_SOURCE_DIR}/../../dfsan_fuzzer_abi.list")

set(CMAKE_CXX_FLAGS_RELEASE
  "${LIBFUZZER_FLAGS_BASE} -O0 -fno-sanitize=all -fsanitize=dataflow -mllvm -sanitizer-coverage-experimental-trace-compares=1 -fsanitize-blacklist=${DFSAN_FUZZER_ABI_LIST}")

foreach(Test ${DFSanTests})
  set_source_files_properties(${Test}.cpp PROPERTIES OBJECT_DEPENDS ${DFSAN_FUZZER_ABI_LIST})
  add_executable(LLVMFuzzer-${Test}
    ${Test}.cpp
    )
  target_link_libraries(LLVMFuzzer-${Test}
    LLVMFuzzer
    )
endforeach()

