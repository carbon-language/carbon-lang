set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(Options.inc
         -gen-opt-parser-defs)
add_custom_target(ClangDriverOptions
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Options.inc)

set(LLVM_TARGET_DEFINITIONS CC1Options.td)
tablegen(CC1Options.inc
         -gen-opt-parser-defs)
add_custom_target(ClangCC1Options
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/CC1Options.inc)
