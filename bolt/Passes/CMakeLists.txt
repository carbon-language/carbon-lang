add_llvm_library(LLVMBOLTPasses
  BinaryPasses.cpp
  FrameOptimizer.cpp
  Inliner.cpp
  ReorderAlgorithm.cpp
  )

include_directories( ${LLVM_MAIN_SRC_DIR}/tools/llvm-bolt )
