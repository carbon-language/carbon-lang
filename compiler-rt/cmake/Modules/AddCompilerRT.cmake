include(AddLLVM)
include(LLVMParseArguments)
include(CompilerRTUtils)

# Tries to add "object library" target for a given architecture
# with name "<name>.<arch>" if architecture can be targeted.
# add_compiler_rt_object_library(<name> <arch>
#                                SOURCES <source files>
#                                CFLAGS <compile flags>)
macro(add_compiler_rt_object_library name arch)
  if(CAN_TARGET_${arch})
    parse_arguments(LIB "SOURCES;CFLAGS" "" ${ARGN})
    add_library(${name}.${arch} OBJECT ${LIB_SOURCES})
    set_target_compile_flags(${name}.${arch}
      ${TARGET_${arch}_CFLAGS} ${LIB_CFLAGS})
  else()
    message(FATAL_ERROR "Archtecture ${arch} can't be targeted")
  endif()
endmacro()

# Adds static runtime for a given architecture and puts it in the proper
# directory in the build and install trees.
# add_compiler_rt_static_runtime(<name> <arch>
#                                SOURCES <source files>
#                                CFLAGS <compile flags>
#                                DEFS <compile definitions>)
macro(add_compiler_rt_static_runtime name arch)
  if(CAN_TARGET_${arch})
    parse_arguments(LIB "SOURCES;CFLAGS;DEFS" "" ${ARGN})
    add_library(${name} STATIC ${LIB_SOURCES})
    # Setup compile flags and definitions.
    set_target_compile_flags(${name}
      ${TARGET_${arch}_CFLAGS} ${LIB_CFLAGS})
    set_property(TARGET ${name} APPEND PROPERTY
      COMPILE_DEFINITIONS ${LIB_DEFS})
    # Setup correct output directory in the build tree.
    set_target_properties(${name} PROPERTIES
            ARCHIVE_OUTPUT_DIRECTORY ${COMPILER_RT_LIBRARY_OUTPUT_DIR})
    # Add installation command.
    install(TARGETS ${name}
      ARCHIVE DESTINATION ${COMPILER_RT_LIBRARY_INSTALL_DIR})
  else()
    message(FATAL_ERROR "Archtecture ${arch} can't be targeted")
  endif()
endmacro()

# Unittests support.
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
