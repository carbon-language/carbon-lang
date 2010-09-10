add_llvm_library(LLVMTarget
  Mangler.cpp
  SubtargetFeature.cpp
  Target.cpp
  TargetAsmLexer.cpp
  TargetData.cpp
  TargetELFWriterInfo.cpp
  TargetFrameInfo.cpp
  TargetInstrInfo.cpp
  TargetIntrinsicInfo.cpp
  TargetLoweringObjectFile.cpp
  TargetMachine.cpp
  TargetRegisterInfo.cpp
  TargetSubtarget.cpp
  )

target_link_libraries(LLVMTarget LLVMCore LLVMMC)
