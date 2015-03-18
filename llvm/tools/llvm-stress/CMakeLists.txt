set(LLVM_LINK_COMPONENTS
  Core
  IPA
  Support
  )

add_llvm_tool(llvm-stress
  llvm-stress.cpp
  )
export_executable_symbols(llvm-stress)
