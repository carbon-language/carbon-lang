set(LLVM_LINK_COMPONENTS
  BitReader
  Core
  Object
  Support
  )

add_llvm_tool(llvm-nm
  llvm-nm.cpp
  )
