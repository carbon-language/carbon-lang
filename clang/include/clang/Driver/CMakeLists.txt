set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(Options.inc
         -gen-opt-parser-defs)
add_custom_target(ClangDriverOptions
  DEPENDS Options.inc)

set(LLVM_TARGET_DEFINITIONS CC1Options.td)
tablegen(CC1Options.inc
         -gen-opt-parser-defs)
add_custom_target(ClangCC1Options
  DEPENDS CC1Options.inc)

set(LLVM_TARGET_DEFINITIONS CC1AsOptions.td)
tablegen(CC1AsOptions.inc
         -gen-opt-parser-defs)
add_custom_target(ClangCC1AsOptions
  DEPENDS CC1AsOptions.inc)
