foreach(c ${LLVM_TARGETS_TO_BUILD})
  set(TARGETS_BUILT "${TARGETS_BUILT} ${c}")
endforeach(c)
set(TARGETS_TO_BUILD ${TARGETS_BUILT})

# FIXME: This won't work for project files, we need to use a --param.
set(LLVM_LIBS_DIR "${LLVM_BINARY_DIR}/lib/${CMAKE_CFG_INTDIR}")
set(SHLIBEXT "${LTDL_SHLIB_EXT}")

set(SHLIBDIR "${LLVM_BINARY_DIR}/lib/${CMAKE_CFG_INTDIR}")

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

set(LIT_ARGS "${LLVM_LIT_ARGS}")
separate_arguments(LIT_ARGS)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/site.exp.in
  ${CMAKE_CURRENT_BINARY_DIR}/site.exp)

MAKE_DIRECTORY(${CMAKE_CURRENT_BINARY_DIR}/Unit)

# Configuration-time: See Unit/lit.site.cfg.in
set(LLVM_BUILD_MODE "%(build_mode)s")

set(LLVM_SOURCE_DIR ${LLVM_MAIN_SRC_DIR})
set(LLVM_BINARY_DIR ${LLVM_BINARY_DIR})
set(LLVM_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}/%(build_config)s")
set(PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE})
set(ENABLE_SHARED ${LLVM_SHARED_LIBS_ENABLED})
set(SHLIBPATH_VAR ${SHLIBPATH_VAR})

if(LLVM_ENABLE_ASSERTIONS AND NOT MSVC_IDE)
  set(ENABLE_ASSERTIONS "1")
else()
  set(ENABLE_ASSERTIONS "0")
endif()

set(HOST_OS ${CMAKE_HOST_SYSTEM_NAME})
set(HOST_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  @ONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  @ONLY)

# Setup the basic dependencies for running LLVM's regression and unit test
# suites.
add_custom_target(check-llvm.deps
  DEPENDS UnitTests
          BugpointPasses LLVMHello
          llc lli llvm-ar llvm-as llvm-dis llvm-extract llvm-dwarfdump
          llvm-link llvm-mc llvm-nm llvm-objdump llvm-readobj
          macho-dump opt
          FileCheck count not
  )
set_target_properties(check-llvm.deps PROPERTIES FOLDER "Tests")

# This is the primary action target to check the LLVM regression and unit test
# suite.
add_custom_target(check-llvm
  COMMAND ${PYTHON_EXECUTABLE}
          ${LLVM_SOURCE_DIR}/utils/lit/lit.py
          --param llvm_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
          --param llvm_unit_site_config=${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
          --param build_config=${CMAKE_CFG_INTDIR}
          --param build_mode=${RUNTIME_BUILD_MODE}
          ${LIT_ARGS}
          ${CMAKE_CURRENT_BINARY_DIR}
          COMMENT "Running LLVM regression tests"
  DEPENDS check-llvm.deps
  )

# Setup a legacy alias for 'check-llvm'. This will likely change to be an
# alias for 'check-all' at some point in the future.
add_custom_target(check DEPENDS check-llvm)
