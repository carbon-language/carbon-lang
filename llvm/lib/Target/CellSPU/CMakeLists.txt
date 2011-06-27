set(LLVM_TARGET_DEFINITIONS SPU.td)

tablegen(SPUGenInstrNames.inc -gen-instr-enums)
tablegen(SPUGenAsmWriter.inc -gen-asm-writer)
tablegen(SPUGenCodeEmitter.inc -gen-emitter)
tablegen(SPUGenRegisterInfo.inc -gen-register-info)
tablegen(SPUGenInstrInfo.inc -gen-instr-desc)
tablegen(SPUGenDAGISel.inc -gen-dag-isel)
tablegen(SPUGenSubtarget.inc -gen-subtarget)
tablegen(SPUGenCallingConv.inc -gen-callingconv)

add_llvm_target(CellSPUCodeGen
  SPUAsmPrinter.cpp
  SPUHazardRecognizers.cpp
  SPUInstrInfo.cpp
  SPUISelDAGToDAG.cpp
  SPUISelLowering.cpp
  SPUFrameLowering.cpp
  SPUMCAsmInfo.cpp
  SPURegisterInfo.cpp
  SPUSubtarget.cpp
  SPUTargetMachine.cpp
  SPUSelectionDAGInfo.cpp
  SPUNopFiller.cpp
  )

add_subdirectory(TargetInfo)
