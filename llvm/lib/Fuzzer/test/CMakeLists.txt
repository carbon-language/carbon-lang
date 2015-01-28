set(Tests
  ExactTest
  InfiniteTest
  NullDerefTest
  SimpleTest
  TimeoutTest
  )

set(TestBinaries)

foreach(Test ${Tests})
  add_executable(LLVMFuzzer-${Test}
    EXCLUDE_FROM_ALL
    ${Test}.cpp
    )
  target_link_libraries(LLVMFuzzer-${Test}
    LLVMFuzzer
    )
  set(TestBinaries ${TestBinaries} LLVMFuzzer-${Test})
endforeach()

set_target_properties(${TestBinaries}
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

set(EXCLUDE_FROM_ALL TRUE)
add_lit_testsuite(check-fuzzer "Running Fuzzer tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${TestBinaries}
    )
set(EXCLUDE_FROM_ALL FALSE)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

