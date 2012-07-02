configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  )

add_lit_testsuite(check-llvm "Running the LLVM regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS llvm_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
         llvm_unit_site_config=${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  DEPENDS BugpointPasses LLVMHello
          llc lli llvm-ar llvm-as llvm-dis llvm-extract llvm-dwarfdump
          llvm-link llvm-mc llvm-nm llvm-objdump llvm-readobj
          macho-dump opt
          FileCheck count not
  )
set_target_properties(check-llvm PROPERTIES FOLDER "Tests")
# Note, this is kept here rather than in the DEPENDS above because of bugs in
# some CMake versions that mishandle the dependency otherwise.
add_dependencies(check-llvm UnitTests)

# Setup a legacy alias for 'check-llvm'. This will likely change to be an
# alias for 'check-all' at some point in the future.
add_custom_target(check DEPENDS check-llvm)
set_target_properties(check PROPERTIES FOLDER "Tests")
