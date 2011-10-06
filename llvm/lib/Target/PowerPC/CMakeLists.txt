set(LLVM_TARGET_DEFINITIONS PPC.td)

llvm_tablegen(PPCGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(PPCGenCodeEmitter.inc -gen-emitter)
llvm_tablegen(PPCGenMCCodeEmitter.inc -gen-emitter -mc-emitter)
llvm_tablegen(PPCGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(PPCGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(PPCGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(PPCGenCallingConv.inc -gen-callingconv)
llvm_tablegen(PPCGenSubtargetInfo.inc -gen-subtarget)
add_public_tablegen_target(PowerPCCommonTableGen)

add_llvm_target(PowerPCCodeGen
  PPCAsmPrinter.cpp
  PPCBranchSelector.cpp
  PPCCodeEmitter.cpp
  PPCHazardRecognizers.cpp
  PPCInstrInfo.cpp
  PPCISelDAGToDAG.cpp
  PPCISelLowering.cpp
  PPCFrameLowering.cpp
  PPCJITInfo.cpp
  PPCMCInstLower.cpp
  PPCRegisterInfo.cpp
  PPCSubtarget.cpp
  PPCTargetMachine.cpp
  PPCSelectionDAGInfo.cpp
  )

add_llvm_library_dependencies(LLVMPowerPCCodeGen
  LLVMAnalysis
  LLVMAsmPrinter
  LLVMCodeGen
  LLVMCore
  LLVMMC
  LLVMPowerPCAsmPrinter
  LLVMPowerPCDesc
  LLVMPowerPCInfo
  LLVMSelectionDAG
  LLVMSupport
  LLVMTarget
  )

add_subdirectory(InstPrinter)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
