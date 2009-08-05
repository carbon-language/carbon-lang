# - macro_add_check_test(test_name test_source linklib1 ... linklibN)

ENABLE_TESTING()
include(CTest)
set(CMAKE_C_FLAGS_PROFILING "-g -pg")

macro (MACRO_ADD_CHECK_TEST _testName _testSource)
  add_executable(${_testName} ${_testSource})
  target_link_libraries(${_testName} ${ARGN})
  add_test(${_testName} ${CMAKE_CURRENT_BINARY_DIR}/${_testName})
endmacro (MACRO_ADD_CHECK_TEST)
