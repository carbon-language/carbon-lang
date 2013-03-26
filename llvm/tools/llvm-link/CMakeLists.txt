set(LLVM_LINK_COMPONENTS linker bitreader bitwriter asmparser irreader)

add_llvm_tool(llvm-link
  llvm-link.cpp
  )
