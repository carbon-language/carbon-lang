set(LLVM_LINK_COMPONENTS
  Support
  IRReader
  BitWriter
  )

add_llvm_tool(llvm-uselistorder
  llvm-uselistorder.cpp
  )
