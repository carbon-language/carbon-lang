set(LLVM_LINK_COMPONENTS support)

add_llvm_tool(llvm-mc
  llvm-mc.cpp
  AsmLexer.cpp
  AsmParser.cpp
  )
