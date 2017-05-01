add_llvm_library(LLVMBOLTPasses
  AllocCombiner.cpp
  BinaryPasses.cpp
  BinaryFunctionCallGraph.cpp
  CallGraph.cpp
  DataflowAnalysis.cpp
  DataflowInfoManager.cpp
  FrameAnalysis.cpp
  FrameOptimizer.cpp
  HFSort.cpp
  HFSortPlus.cpp
  IndirectCallPromotion.cpp
  Inliner.cpp
  LivenessAnalysis.cpp
  PettisAndHansen.cpp
  ReorderAlgorithm.cpp
  ReorderFunctions.cpp
  ShrinkWrapping.cpp
  StackAllocationAnalysis.cpp
  StackAvailableExpressions.cpp
  StackPointerTracking.cpp
  StackReachingUses.cpp
  )

include_directories( ${LLVM_MAIN_SRC_DIR}/tools/llvm-bolt )
