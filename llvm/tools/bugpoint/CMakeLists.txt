set(LLVM_LINK_COMPONENTS asmparser instrumentation scalaropts ipo
  linker bitreader bitwriter irreader vectorize objcarcopts)

add_llvm_tool(bugpoint
  BugDriver.cpp
  CrashDebugger.cpp
  ExecutionDriver.cpp
  ExtractFunction.cpp
  FindBugs.cpp
  Miscompilation.cpp
  OptimizerDriver.cpp
  ToolRunner.cpp
  bugpoint.cpp
  )
set_target_properties(bugpoint PROPERTIES ENABLE_EXPORTS 1)
