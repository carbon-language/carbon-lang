include(AddLLVM)
include(LLVMParseArguments)

set(COMPILER_RT_GTEST_PATH ${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest)
set(COMPILER_RT_GTEST_SOURCE ${COMPILER_RT_GTEST_PATH}/gtest-all.cc)
set(COMPILER_RT_GTEST_INCLUDE_CFLAGS
  -DGTEST_NO_LLVM_RAW_OSTREAM=1
  -I${COMPILER_RT_GTEST_PATH}/include
)

# Use Clang to link objects into a single executable with just-built
# Clang, using specific link flags. Make executable a part of provided
# test_suite.
# add_compiler_rt_test(<test_suite> <test_name>
#                      OBJECTS <object files>
#                      DEPS <deps (e.g. runtime libs)>
#                      LINK_FLAGS <link flags>)
macro(add_compiler_rt_test test_suite test_name)
  parse_arguments(TEST "OBJECTS;DEPS;LINK_FLAGS" "" ${ARGN})
  get_unittest_directory(OUTPUT_DIR)
  file(MAKE_DIRECTORY ${OUTPUT_DIR})
  set(output_bin "${OUTPUT_DIR}/${test_name}")
  add_custom_command(
    OUTPUT ${output_bin}
    COMMAND clang ${TEST_OBJECTS} -o "${output_bin}"
            ${TEST_LINK_FLAGS}
    DEPENDS clang ${TEST_DEPS})
  add_custom_target(${test_name} DEPENDS ${output_bin})
  # Make the test suite depend on the binary.
  add_dependencies(${test_suite} ${test_name})
endmacro()
