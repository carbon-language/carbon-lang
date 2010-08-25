set(LLVM_LINK_COMPONENTS linker bitreader bitwriter asmparser transformutils)

add_llvm_tool(llvm-link
  llvm-link.cpp
  )
