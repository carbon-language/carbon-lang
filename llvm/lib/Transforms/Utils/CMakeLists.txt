add_llvm_library(LLVMTransformUtils
  AddrModeMatcher.cpp
  BasicBlockUtils.cpp
  BasicInliner.cpp
  BreakCriticalEdges.cpp
  CloneFunction.cpp
  CloneLoop.cpp
  CloneModule.cpp
  CodeExtractor.cpp
  DemoteRegToStack.cpp
  InlineFunction.cpp
  InstructionNamer.cpp
  LCSSA.cpp
  Local.cpp
  LoopSimplify.cpp
  LoopUnroll.cpp
  LowerInvoke.cpp
  LowerSwitch.cpp
  Mem2Reg.cpp
  PromoteMemoryToRegister.cpp
  SSAUpdater.cpp
  SSI.cpp
  SimplifyCFG.cpp
  UnifyFunctionExitNodes.cpp
  ValueMapper.cpp
  )

target_link_libraries (LLVMTransformUtils LLVMSupport)
