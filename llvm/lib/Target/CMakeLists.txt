add_llvm_library(LLVMTarget
  Mangler.cpp
  Target.cpp
  TargetData.cpp
  TargetELFWriterInfo.cpp
  TargetFrameLowering.cpp
  TargetInstrInfo.cpp
  TargetIntrinsicInfo.cpp
  TargetLibraryInfo.cpp
  TargetLoweringObjectFile.cpp
  TargetMachine.cpp
  TargetRegisterInfo.cpp
  TargetSubtargetInfo.cpp
  )

add_llvm_library_dependencies(LLVMTarget
  LLVMCore
  LLVMMC
  LLVMSupport
  )

foreach(t ${LLVM_TARGETS_TO_BUILD})
  message(STATUS "Targeting ${t}")
  add_subdirectory(${t})
endforeach()
