set(LLVM_TARGET_DEFINITIONS FIROps.td)
mlir_tablegen(FIROps.h.inc -gen-op-decls)
mlir_tablegen(FIROps.cpp.inc -gen-op-defs)
add_public_tablegen_target(FIROpsIncGen)
