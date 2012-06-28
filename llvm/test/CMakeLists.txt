set(LIT_ARGS "${LLVM_LIT_ARGS}")
separate_arguments(LIT_ARGS)

MAKE_DIRECTORY(${CMAKE_CURRENT_BINARY_DIR}/Unit)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  )

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
