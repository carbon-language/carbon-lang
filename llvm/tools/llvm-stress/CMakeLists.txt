set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  Support
  )

add_llvm_tool(llvm-stress
  llvm-stress.cpp
  )
export_executable_symbols(llvm-stress)
