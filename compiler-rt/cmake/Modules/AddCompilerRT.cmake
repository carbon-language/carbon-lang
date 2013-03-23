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

# Same as above, but adds universal osx library with name "<name>.osx"
# targeting multiple architectures.
# add_compiler_rt_osx_object_library(<name> ARCH <architectures>
#                                           SOURCES <source files>
#                                           CFLAGS <compile flags>)
macro(add_compiler_rt_osx_object_library name)
  parse_arguments(LIB "ARCH;SOURCES;CFLAGS" "" ${ARGN})
  set(libname "${name}.osx")
  add_library(${libname} OBJECT ${LIB_SOURCES})
  set_target_compile_flags(${libname} ${LIB_CFLAGS})
  set_target_properties(${libname} PROPERTIES OSX_ARCHITECTURES "${LIB_ARCH}")
endmacro()

# Adds static runtime for a given architecture and puts it in the proper
# directory in the build and install trees.
# add_compiler_rt_static_runtime(<name> <arch>
#                                SOURCES <source files>
#                                CFLAGS <compile flags>
#                                DEFS <compile definitions>
#                                SYMS <symbols file>)
macro(add_compiler_rt_static_runtime name arch)
  if(CAN_TARGET_${arch})
    parse_arguments(LIB "SOURCES;CFLAGS;DEFS;SYMS" "" ${ARGN})
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
    # Generate the .syms file if possible.
    if(LIB_SYMS)
      get_target_property(libfile ${name} LOCATION)
      configure_file(${LIB_SYMS} ${libfile}.syms)
      install(FILES ${libfile}.syms
        DESTINATION ${COMPILER_RT_LIBRARY_INSTALL_DIR})
    endif(LIB_SYMS)
  else()
    message(FATAL_ERROR "Archtecture ${arch} can't be targeted")
  endif()
endmacro()

# Same as add_compiler_rt_static_runtime, but creates a universal library
# for several architectures.
# add_compiler_rt_osx_static_runtime(<name> ARCH <architectures>
#                                    SOURCES <source files>
#                                    CFLAGS <compile flags>
#                                    DEFS <compile definitions>)
macro(add_compiler_rt_osx_static_runtime name)
  parse_arguments(LIB "ARCH;SOURCES;CFLAGS;DEFS" "" ${ARGN})
  add_library(${name} STATIC ${LIB_SOURCES})
  set_target_compile_flags(${name} ${LIB_CFLAGS})
  set_property(TARGET ${name} APPEND PROPERTY
    COMPILE_DEFINITIONS ${LIB_DEFS})
  set_target_properties(${name} PROPERTIES
    OSX_ARCHITECTURES "${LIB_ARCH}"
    ARCHIVE_OUTPUT_DIRECTORY ${COMPILER_RT_LIBRARY_OUTPUT_DIR})
  install(TARGETS ${name}
    ARCHIVE DESTINATION ${COMPILER_RT_LIBRARY_INSTALL_DIR})
endmacro()

# Adds dynamic runtime library on osx, which supports multiple architectures.
# add_compiler_rt_osx_dynamic_runtime(<name> ARCH <architectures>
#                                     SOURCES <source files>
#                                     CFLAGS <compile flags>
#                                     DEFS <compile definitions>
#                                     LINKFLAGS <link flags>)
macro(add_compiler_rt_osx_dynamic_runtime name)
  parse_arguments(LIB "ARCH;SOURCES;CFLAGS;DEFS;LINKFLAGS" "" ${ARGN})
  add_library(${name} SHARED ${LIB_SOURCES})
  set_target_compile_flags(${name} ${LIB_CFLAGS})
  set_target_link_flags(${name} ${LIB_LINKFLAGS})
  set_property(TARGET ${name} APPEND PROPERTY
    COMPILE_DEFINITIONS ${LIB_DEFS})
  set_target_properties(${name} PROPERTIES
    OSX_ARCHITECTURES "${LIB_ARCH}"
    LIBRARY_OUTPUT_DIRECTORY ${COMPILER_RT_LIBRARY_OUTPUT_DIR})
  install(TARGETS ${name}
    LIBRARY DESTINATION ${COMPILER_RT_LIBRARY_INSTALL_DIR})
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
  set(output_bin "${CMAKE_CURRENT_BINARY_DIR}/${test_name}")
  add_custom_target(${test_name}
    COMMAND clang ${TEST_OBJECTS} -o "${output_bin}"
            ${TEST_LINK_FLAGS}
    DEPENDS clang ${TEST_DEPS})
  # Make the test suite depend on the binary.
  add_dependencies(${test_suite} ${test_name})
endmacro()
