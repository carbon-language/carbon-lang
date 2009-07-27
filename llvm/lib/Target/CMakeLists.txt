add_llvm_library(LLVMTarget
  COFFTargetAsmInfo.cpp
  DarwinTargetAsmInfo.cpp
  ELFTargetAsmInfo.cpp
  SubtargetFeature.cpp
  Target.cpp
  TargetAsmInfo.cpp
  TargetData.cpp
  TargetELFWriterInfo.cpp
  TargetFrameInfo.cpp
  TargetInstrInfo.cpp
  TargetIntrinsicInfo.cpp
  TargetMachOWriterInfo.cpp
  TargetMachine.cpp
  TargetRegisterInfo.cpp
  TargetSubtarget.cpp
  )

# TODO: Support other targets besides X86. See Makefile.
