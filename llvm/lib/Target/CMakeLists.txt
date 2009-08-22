add_llvm_library(LLVMTarget
  SubtargetFeature.cpp
  Target.cpp
  TargetData.cpp
  TargetELFWriterInfo.cpp
  TargetFrameInfo.cpp
  TargetInstrInfo.cpp
  TargetIntrinsicInfo.cpp
  TargetLoweringObjectFile.cpp
  TargetMachOWriterInfo.cpp
  TargetMachine.cpp
  TargetRegisterInfo.cpp
  TargetSubtarget.cpp
  )

# TODO: Support other targets besides X86. See Makefile.
