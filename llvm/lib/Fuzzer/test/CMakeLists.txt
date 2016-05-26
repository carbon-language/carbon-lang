# Build all these tests with -O0, otherwise optimizations may merge some
# basic blocks and we'll fail to discover the targets.
# We change the flags for every build type because we might be doing
# a multi-configuration build (e.g. Xcode) where CMAKE_BUILD_TYPE doesn't
# mean anything.
set(variables_to_filter
  CMAKE_CXX_FLAGS_RELEASE
  CMAKE_CXX_FLAGS_DEBUG
  CMAKE_CXX_FLAGS_RELWITHDEBINFO
  CMAKE_CXX_FLAGS_MINSIZEREL
  LIBFUZZER_FLAGS_BASE
  )
foreach (VARNAME ${variables_to_filter})
  string(REPLACE " " ";" BUILD_FLAGS_AS_LIST "${${VARNAME}}")
  set(new_flags "")
  foreach (flag ${BUILD_FLAGS_AS_LIST})
    # NOTE: Use of XX here is to avoid a CMake warning due to CMP0054
    if (NOT ("XX${flag}" MATCHES "XX-O[0123s]"))
      set(new_flags "${new_flags} ${flag}")
    else()
      set(new_flags "${new_flags} -O0")
    endif()
  endforeach()
  set(${VARNAME} "${new_flags}")
endforeach()

# Enable the coverage instrumentation (it is disabled for the Fuzzer lib).
set(CMAKE_CXX_FLAGS "${LIBFUZZER_FLAGS_BASE} -fsanitize-coverage=edge,indirect-calls")

set(DFSanTests
  MemcmpTest
  SimpleCmpTest
  StrcmpTest
  StrncmpTest
  SwitchTest
  )

set(Tests
  AccumulateAllocationsTest
  BufferOverflowOnInput
  CallerCalleeTest
  CounterTest
  CustomMutatorTest
  EmptyTest
  FourIndependentBranchesTest
  FullCoverageSetTest
  InitializeTest
  MemcmpTest
  LeakTest
  LeakTimeoutTest
  NullDerefTest
  NullDerefOnEmptyTest
  NthRunCrashTest
  OutOfMemoryTest
  RepeatedMemcmp
  SimpleCmpTest
  SimpleDictionaryTest
  SimpleFnAdapterTest
  SimpleHashTest
  SimpleTest
  SpamyTest
  StrcmpTest
  StrncmpTest
  SwitchTest
  ThreadedTest
  TimeoutTest
  )

set(CustomMainTests
  )

set(UninstrumentedTests
  UninstrumentedTest
  )

set(TraceBBTests
  SimpleTest
  )

set(TracePCTests
  FourIndependentBranchesTest
  FullCoverageSetTest
  )

set(UbsanTests
  SignedIntOverflowTest
  )

set(TestBinaries)

foreach(Test ${Tests})
  add_executable(LLVMFuzzer-${Test}
    ${Test}.cpp
    )
  target_link_libraries(LLVMFuzzer-${Test}
    LLVMFuzzer
    )
  set(TestBinaries ${TestBinaries} LLVMFuzzer-${Test})
endforeach()

foreach(Test ${CustomMainTests})
  add_executable(LLVMFuzzer-${Test}
    ${Test}.cpp
    )
  target_link_libraries(LLVMFuzzer-${Test}
    LLVMFuzzerNoMain
    )
  set(TestBinaries ${TestBinaries} LLVMFuzzer-${Test})
endforeach()


configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/unit/lit.site.cfg
  )

include_directories(..)
include_directories(${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include)

add_executable(LLVMFuzzer-Unittest
  FuzzerUnittest.cpp
  FuzzerFnAdapterUnittest.cpp
  $<TARGET_OBJECTS:LLVMFuzzerNoMainObjects>
  )

target_link_libraries(LLVMFuzzer-Unittest
  gtest
  gtest_main
  )

set(TestBinaries ${TestBinaries} LLVMFuzzer-Unittest)

add_subdirectory(dfsan)

foreach(Test ${DFSanTests})
  set(TestBinaries ${TestBinaries} LLVMFuzzer-${Test}-DFSan)
endforeach()

add_subdirectory(uninstrumented)

foreach(Test ${UninstrumentedTests})
  set(TestBinaries ${TestBinaries} LLVMFuzzer-${Test}-Uninstrumented)
endforeach()

add_subdirectory(ubsan)

foreach(Test ${UbsanTests})
  set(TestBinaries ${TestBinaries} LLVMFuzzer-${Test}-Ubsan)
endforeach()

add_subdirectory(trace-bb)

foreach(Test ${TraceBBTests})
  set(TestBinaries ${TestBinaries} LLVMFuzzer-${Test}-TraceBB)
endforeach()

add_subdirectory(trace-pc)

foreach(Test ${TracePCTests})
  set(TestBinaries ${TestBinaries} LLVMFuzzer-${Test}-TracePC)
endforeach()

set_target_properties(${TestBinaries}
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

add_lit_testsuite(check-fuzzer "Running Fuzzer tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${TestBinaries} FileCheck not
    )
