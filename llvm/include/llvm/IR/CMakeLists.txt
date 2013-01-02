set(LLVM_TARGET_DEFINITIONS Intrinsics.td)

tablegen(LLVM Intrinsics.gen -gen-intrinsic)

add_custom_target(intrinsics_gen ALL
  DEPENDS ${llvm_builded_incs_dir}/IR/Intrinsics.gen)
set_target_properties(intrinsics_gen PROPERTIES FOLDER "Tablegenning")
