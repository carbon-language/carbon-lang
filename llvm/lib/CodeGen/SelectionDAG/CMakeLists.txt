add_llvm_library(LLVMSelectionDAG
  CallingConvLower.cpp
  DAGCombiner.cpp
  FastISel.cpp
  InstrEmitter.cpp
  LegalizeDAG.cpp
  LegalizeFloatTypes.cpp
  LegalizeIntegerTypes.cpp
  LegalizeTypes.cpp
  LegalizeTypesGeneric.cpp
  LegalizeVectorOps.cpp
  LegalizeVectorTypes.cpp
  ScheduleDAGFast.cpp
  ScheduleDAGList.cpp
  ScheduleDAGRRList.cpp
  ScheduleDAGSDNodes.cpp
  SelectionDAG.cpp
  SelectionDAGBuild.cpp
  SelectionDAGISel.cpp
  SelectionDAGPrinter.cpp
  TargetLowering.cpp
  )

target_link_libraries (LLVMSelectionDAG LLVMAnalysis LLVMAsmPrinter LLVMCodeGen)
