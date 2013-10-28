configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  )

# Don't include check-llvm into check-all without LLVM_BUILD_TOOLS.
if(NOT LLVM_BUILD_TOOLS)
  set(EXCLUDE_FROM_ALL ON)
endif()

# Set the depends list as a variable so that it can grow conditionally.
# NOTE: Sync the substitutions in test/lit.cfg when adding to this list.
set(LLVM_TEST_DEPENDS
          UnitTests
          BugpointPasses
          LLVMHello
          llc
          lli
          lli-child-target
          llvm-ar
          llvm-as
          llvm-bcanalyzer
          llvm-c-test
          llvm-cov
          llvm-diff
          llvm-dis
          llvm-extract
          llvm-dwarfdump
          llvm-link
          llvm-lto
          llvm-mc
          llvm-mcmarkup
          llvm-nm
          llvm-objdump
          llvm-readobj
          llvm-rtdyld
          llvm-symbolizer
          macho-dump
          opt
          profile_rt-shared
          FileCheck
          count
          not
          yaml2obj
          obj2yaml
        )

# If Intel JIT events are supported, depend on a tool that tests the listener.
if( LLVM_USE_INTEL_JITEVENTS )
  set(LLVM_TEST_DEPENDS ${LLVM_TEST_DEPENDS} llvm-jitlistener)
endif( LLVM_USE_INTEL_JITEVENTS )

add_lit_testsuite(check-llvm "Running the LLVM regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS llvm_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
         llvm_unit_site_config=${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  DEPENDS ${LLVM_TEST_DEPENDS}
  )
set_target_properties(check-llvm PROPERTIES FOLDER "Tests")

# Setup a legacy alias for 'check-llvm'. This will likely change to be an
# alias for 'check-all' at some point in the future.
add_custom_target(check)
add_dependencies(check check-llvm)
set_target_properties(check PROPERTIES FOLDER "Tests")
