foreach(c ${LLVM_TARGETS_TO_BUILD})
  set(TARGETS_BUILT "${TARGETS_BUILT} ${c}")
endforeach(c)
set(TARGETS_TO_BUILD ${TARGETS_BUILT})

# FIXME: This won't work for project files, we need to use a --param.
set(LLVM_LIBS_DIR "${LLVM_BINARY_DIR}/lib/${CMAKE_CFG_INTDIR}")
set(SHLIBEXT "${LTDL_SHLIB_EXT}")

if(BUILD_SHARED_LIBS)
  set(LLVM_SHARED_LIBS_ENABLED "1")
else()
  set(LLVM_SHARED_LIBS_ENABLED "0")
endif(BUILD_SHARED_LIBS)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(SHLIBPATH_VAR "DYLD_LIBRARY_PATH")
else() # Default for all other unix like systems.
  # CMake hardcodes the library locaction using rpath.
  # Therefore LD_LIBRARY_PATH is not required to run binaries in the
  # build dir. We pass it anyways.
  set(SHLIBPATH_VAR "LD_LIBRARY_PATH")
endif()

include(FindPythonInterp)
if(PYTHONINTERP_FOUND)
  set(LIT_ARGS "${LLVM_LIT_ARGS}")
  separate_arguments(LIT_ARGS)

  get_directory_property(DEFINITIONS COMPILE_DEFINITIONS)
  foreach(DEF ${DEFINITIONS})
    set(DEFS "${DEFS} -D${DEF}")
  endforeach()
  get_directory_property(INC_DIRS INCLUDE_DIRECTORIES)
  foreach(INC_DIR ${INC_DIRS})
    set(IDIRS "${IDIRS} -I${INC_DIR}")
  endforeach()
  string(REPLACE "<CMAKE_CXX_COMPILER>" "${CMAKE_CXX_COMPILER}" TEST_COMPILE_CXX_CMD ${CMAKE_CXX_COMPILE_OBJECT})
  string(REPLACE "<DEFINES>"            "${DEFS}"               TEST_COMPILE_CXX_CMD ${TEST_COMPILE_CXX_CMD})
  string(REPLACE "<FLAGS>"              "${CMAKE_CXX_FLAGS}"    TEST_COMPILE_CXX_CMD ${TEST_COMPILE_CXX_CMD})
  string(REPLACE "-o"                   ""                      TEST_COMPILE_CXX_CMD ${TEST_COMPILE_CXX_CMD})
  string(REGEX REPLACE "<[^>]+>"        ""                      TEST_COMPILE_CXX_CMD ${TEST_COMPILE_CXX_CMD})
  set(TEST_COMPILE_CXX_CMD "${TEST_COMPILE_CXX_CMD} ${IDIRS}")
  if(NOT MSVC)
    set(TEST_COMPILE_CXX_CMD "${TEST_COMPILE_CXX_CMD} -x c++")
  endif()
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/site.exp.in
    ${CMAKE_CURRENT_BINARY_DIR}/site.exp)

  MAKE_DIRECTORY(${CMAKE_CURRENT_BINARY_DIR}/Unit)

  set(LLVM_SOURCE_DIR ${LLVM_MAIN_SRC_DIR})
  set(LLVM_BINARY_DIR ${LLVM_BINARY_DIR})
  set(LLVM_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}/${CMAKE_CFG_INTDIR}")
  set(LLVMGCCDIR "")
  set(PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE})
  if (CMAKE_CONFIGURATION_TYPES)
    # FIXME: We have no idea. It could be any of them... So just output all of
    # them.
    set(LLVM_BUILD_MODE "${CMAKE_CONFIGURATION_TYPES}")
  elseif (CMAKE_BUILD_TYPE)
    set(LLVM_BUILD_MODE "${CMAKE_BUILD_TYPE}")
  else()
    set(LLVM_BUILD_MODE "None")
  endif()
  set(ENABLE_SHARED ${LLVM_SHARED_LIBS_ENABLED})
  set(SHLIBPATH_VAR ${SHLIBPATH_VAR})

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
    @ONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
    @ONLY)

  add_custom_target(check
    COMMAND ${PYTHON_EXECUTABLE}
                ${LLVM_SOURCE_DIR}/utils/lit/lit.py
                --param llvm_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
                --param llvm_unit_site_config=${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
                ${LIT_ARGS}
                ${CMAKE_CURRENT_BINARY_DIR}
                DEPENDS
                COMMENT "Running LLVM regression tests")

endif()
