set(LLVM_LINK_COMPONENTS mcjit jit interpreter nativecodegen bitreader asmparser selectiondag)

add_llvm_tool(lli
  lli.cpp
  )
