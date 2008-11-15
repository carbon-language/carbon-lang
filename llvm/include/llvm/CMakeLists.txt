set(LLVM_TARGET_DEFINITIONS Intrinsics.td)

tablegen(Intrinsics.gen -gen-intrinsic)

add_custom_target(intrinsics_gen ALL
  DEPENDS ${llvm_builded_incs_dir}/Intrinsics.gen)

set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} intrinsics_gen PARENT_SCOPE)
