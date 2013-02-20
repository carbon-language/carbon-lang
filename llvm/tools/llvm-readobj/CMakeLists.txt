set(LLVM_LINK_COMPONENTS archive bitreader object)

add_llvm_tool(llvm-readobj
  ELF.cpp
  llvm-readobj.cpp
  )
