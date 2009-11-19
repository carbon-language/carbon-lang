set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(Options.inc
         -gen-opt-parser-defs)
add_custom_target(ClangDriverOptions
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Options.inc)
