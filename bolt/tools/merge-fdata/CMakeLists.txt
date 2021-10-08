set(LLVM_LINK_COMPONENTS Support)

add_llvm_tool(merge-fdata
  merge-fdata.cpp

  DEPENDS
  intrinsics_gen
)
