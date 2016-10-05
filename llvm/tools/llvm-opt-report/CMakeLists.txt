set(LLVM_LINK_COMPONENTS Core Object Support)

add_llvm_tool(llvm-opt-report
  OptReport.cpp
  )

