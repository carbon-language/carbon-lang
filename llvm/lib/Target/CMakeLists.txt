add_llvm_library(LLVMTarget
  DarwinTargetAsmInfo.cpp
  ELFTargetAsmInfo.cpp
  SubtargetFeature.cpp
  Target.cpp
  TargetAsmInfo.cpp
  TargetData.cpp
  TargetELFWriterInfo.cpp
  TargetFrameInfo.cpp
  TargetInstrInfo.cpp
  TargetMachOWriterInfo.cpp
  TargetMachine.cpp
  TargetRegisterInfo.cpp
  TargetSubtarget.cpp
  )

# TODO: Support other targets besides X86. See Makefile.
