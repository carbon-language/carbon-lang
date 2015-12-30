

macro(internal_find_llvm_parts)
# Rely on llvm-config.
  set(CONFIG_OUTPUT)
  find_program(LLVM_CONFIG "llvm-config")
  if(DEFINED LLVM_PATH)
    set(LLVM_INCLUDE_DIR ${LLVM_INCLUDE_DIR} CACHE PATH "Path to llvm/include")
    set(LLVM_PATH ${LLVM_PATH} CACHE PATH "Path to LLVM source tree")
    set(LLVM_MAIN_SRC_DIR ${LLVM_PATH})
    set(LLVM_CMAKE_PATH "${LLVM_PATH}/cmake/modules")
  elseif(LLVM_CONFIG)
    message(STATUS "Found LLVM_CONFIG as ${LLVM_CONFIG}")
    set(CONFIG_COMMAND ${LLVM_CONFIG}
      "--includedir"
      "--prefix"
      "--src-root")
    execute_process(
      COMMAND ${CONFIG_COMMAND}
      RESULT_VARIABLE HAD_ERROR
      OUTPUT_VARIABLE CONFIG_OUTPUT
    )
    if(NOT HAD_ERROR)
      string(REGEX REPLACE
        "[ \t]*[\r\n]+[ \t]*" ";"
        CONFIG_OUTPUT ${CONFIG_OUTPUT})
    else()
      string(REPLACE ";" " " CONFIG_COMMAND_STR "${CONFIG_COMMAND}")
      message(STATUS "${CONFIG_COMMAND_STR}")
      message(FATAL_ERROR "llvm-config failed with status ${HAD_ERROR}")
    endif()

    list(GET CONFIG_OUTPUT 0 INCLUDE_DIR)
    list(GET CONFIG_OUTPUT 1 LLVM_OBJ_ROOT)
    list(GET CONFIG_OUTPUT 2 MAIN_SRC_DIR)

    set(LLVM_INCLUDE_DIR ${INCLUDE_DIR} CACHE PATH "Path to llvm/include")
    set(LLVM_BINARY_DIR ${LLVM_OBJ_ROOT} CACHE PATH "Path to LLVM build tree")
    set(LLVM_MAIN_SRC_DIR ${MAIN_SRC_DIR} CACHE PATH "Path to LLVM source tree")
    set(LLVM_CMAKE_PATH "${LLVM_BINARY_DIR}/share/llvm/cmake")
  else()
    set(LLVM_FOUND OFF)
    return()
  endif()

  if (NOT EXISTS ${LLVM_MAIN_SRC_DIR})
    set(LLVM_FOUND OFF)
    message(WARNING "Not found: ${LLVM_MAIN_SRC_DIR}")
    return()
  endif()

  if(NOT EXISTS ${LLVM_CMAKE_PATH})
    set(LLVM_FOUND OFF)
    message(WARNING "Not found: ${LLVM_CMAKE_PATH}")
    return()
  endif()

  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_PATH}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_MAIN_SRC_DIR}/cmake/modules")

  set(LLVM_FOUND ON)
endmacro(internal_find_llvm_parts)


macro(internal_simulate_llvm_options)
  # LLVM Options --------------------------------------------------------------
  # Configure the LLVM CMake options expected by libc++.

  include(FindPythonInterp)
  if( NOT PYTHONINTERP_FOUND )
    message(WARNING "Failed to find python interpreter. "
                    "The libc++ test suite will be disabled.")
    set(LLVM_INCLUDE_TESTS OFF)
  endif()

  if (NOT DEFINED LLVM_INCLUDE_TESTS)
    set(LLVM_INCLUDE_TESTS ${LLVM_FOUND})
  endif()
  if (NOT DEFINED LLVM_INCLUDE_DOCS)
    set(LLVM_INCLUDE_DOCS ${LLVM_FOUND})
  endif()
  if (NOT DEFINED LLVM_ENABLE_SPHINX)
    set(LLVM_ENABLE_SPHINX OFF)
  endif()

  # Required LIT Configuration ------------------------------------------------
  # Define the default arguments to use with 'lit', and an option for the user
  # to override.
  set(LIT_ARGS_DEFAULT "-sv --show-xfail --show-unsupported")
  if (MSVC OR XCODE)
    set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
  endif()
  set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

  # Make sure we can use the console pool for recent cmake and ninja > 1.5
  # Needed for add_lit_testsuite
  if(CMAKE_VERSION VERSION_LESS 3.1.20141117)
    set(cmake_3_2_USES_TERMINAL)
  else()
    set(cmake_3_2_USES_TERMINAL USES_TERMINAL)
  endif()

  # Required doc configuration
  if (LLVM_ENABLE_SPHINX)
    message(STATUS "Sphinx enabled.")
    find_package(Sphinx REQUIRED)
  else()
    message(STATUS "Sphinx disabled.")
  endif()

  # FIXME - This is cribbed from HandleLLVMOptions.cmake.
  if(WIN32)
    set(LLVM_HAVE_LINK_VERSION_SCRIPT 0)
    if(CYGWIN)
      set(LLVM_ON_WIN32 0)
      set(LLVM_ON_UNIX 1)
    else(CYGWIN)
      set(LLVM_ON_WIN32 1)
      set(LLVM_ON_UNIX 0)
    endif(CYGWIN)
  else(WIN32)
    if(UNIX)
      set(LLVM_ON_WIN32 0)
      set(LLVM_ON_UNIX 1)
      if(APPLE)
        set(LLVM_HAVE_LINK_VERSION_SCRIPT 0)
      else(APPLE)
        set(LLVM_HAVE_LINK_VERSION_SCRIPT 1)
      endif(APPLE)
    else(UNIX)
      MESSAGE(SEND_ERROR "Unable to determine platform")
    endif(UNIX)
  endif(WIN32)
endmacro(internal_simulate_llvm_options)


macro(handle_out_of_tree_llvm)
  # This macro should not be called unless we are building out of tree.
  # Enforce that.
  if (NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "libc++ incorrectly configured for out-of-tree LLVM")
  endif()

  # Attempt to find an LLVM installation and source directory. Warn if they
  # are not found.
  internal_find_llvm_parts()
  if (NOT LLVM_FOUND)
    message(WARNING "UNSUPPORTED LIBCXX CONFIGURATION DETECTED: "
                    "llvm-config not found and LLVM_PATH not defined.\n"
                    "Reconfigure with -DLLVM_CONFIG=path/to/llvm-config "
                    "or -DLLVM_PATH=path/to/llvm-source-root.")
  endif()

  # Simulate the LLVM CMake options and variables provided by an in-tree LLVM.
  internal_simulate_llvm_options()

  # Additionally include the LLVM CMake functions if we can find the module.
  include(AddLLVM OPTIONAL)
endmacro()
