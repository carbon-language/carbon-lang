set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(CLANG Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(ClangDriverOptions)

set(LLVM_TARGET_DEFINITIONS CC1AsOptions.td)
tablegen(CLANG CC1AsOptions.inc -gen-opt-parser-defs)
add_public_tablegen_target(ClangCC1AsOptions)
