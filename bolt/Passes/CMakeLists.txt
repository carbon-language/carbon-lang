add_llvm_library(LLVMBOLTPasses
  BinaryPasses.cpp
  DataflowAnalysis.cpp
  DataflowInfoManager.cpp
  FrameAnalysis.cpp
  FrameOptimizer.cpp
  HFSort.cpp
  HFSortPlus.cpp
  IndirectCallPromotion.cpp
  Inliner.cpp
  LivenessAnalysis.cpp
  ReorderAlgorithm.cpp
  StackPointerTracking.cpp
  )

include_directories( ${LLVM_MAIN_SRC_DIR}/tools/llvm-bolt )
