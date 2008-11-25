set(LLVM_LINK_COMPONENTS support system)
set(LLVM_REQUIRES_EH 1)

add_llvm_tool(llvmc2
  Action.cpp
  CompilationGraph.cpp
  llvmc.cpp
  Plugin.cpp
  )
