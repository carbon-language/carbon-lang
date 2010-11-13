set(LLVM_LINK_COMPONENTS jit interpreter nativecodegen bitreader asmparser selectiondag)

add_llvm_tool(lli
  lli.cpp
  )
