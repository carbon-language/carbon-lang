set(LLVM_LINK_COMPONENTS
  AsmParser
  BitReader
  BitWriter
  Core
  IRReader
  Support
  )

add_llvm_tool(llvm-uselistorder
  llvm-uselistorder.cpp
  )
