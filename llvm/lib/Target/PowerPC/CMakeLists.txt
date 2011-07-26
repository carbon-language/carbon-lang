set(LLVM_TARGET_DEFINITIONS PPC.td)

tablegen(PPCGenAsmWriter.inc -gen-asm-writer)
tablegen(PPCGenCodeEmitter.inc -gen-emitter)
tablegen(PPCGenMCCodeEmitter.inc -gen-emitter -mc-emitter)
tablegen(PPCGenRegisterInfo.inc -gen-register-info)
tablegen(PPCGenInstrInfo.inc -gen-instr-info)
tablegen(PPCGenDAGISel.inc -gen-dag-isel)
tablegen(PPCGenCallingConv.inc -gen-callingconv)
tablegen(PPCGenSubtargetInfo.inc -gen-subtarget)
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

add_subdirectory(InstPrinter)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
