set(LLVM_TARGET_DEFINITIONS Intrinsics.td)

tablegen(LLVM Intrinsics.gen -gen-intrinsic)

add_custom_target(intrinsics_gen ALL
  DEPENDS ${LLVM_INCLUDE_DIR}/llvm/IR/Intrinsics.gen)
set_target_properties(intrinsics_gen PROPERTIES FOLDER "Tablegenning")
