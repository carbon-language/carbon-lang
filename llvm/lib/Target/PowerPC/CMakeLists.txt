set(LLVM_TARGET_DEFINITIONS PPC.td)

tablegen(PPCGenAsmWriter.inc -gen-asm-writer)
tablegen(PPCGenCodeEmitter.inc -gen-emitter)
tablegen(PPCGenMCCodeEmitter.inc -gen-emitter -mc-emitter)
tablegen(PPCGenRegisterInfo.inc -gen-register-info)
tablegen(PPCGenInstrInfo.inc -gen-instr-info)
tablegen(PPCGenDAGISel.inc -gen-dag-isel)
tablegen(PPCGenCallingConv.inc -gen-callingconv)
tablegen(PPCGenSubtargetInfo.inc -gen-subtarget)

add_llvm_target(PowerPCCodeGen
  PPCAsmBackend.cpp
  PPCAsmPrinter.cpp
  PPCBranchSelector.cpp
  PPCCodeEmitter.cpp
  PPCHazardRecognizers.cpp
  PPCInstrInfo.cpp
  PPCISelDAGToDAG.cpp
  PPCISelLowering.cpp
  PPCFrameLowering.cpp
  PPCJITInfo.cpp
  PPCMCCodeEmitter.cpp
  PPCMCInstLower.cpp
  PPCPredicates.cpp
  PPCRegisterInfo.cpp
  PPCSubtarget.cpp
  PPCTargetMachine.cpp
  PPCSelectionDAGInfo.cpp
  )

add_subdirectory(InstPrinter)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
