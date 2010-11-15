set(LLVM_TARGET_DEFINITIONS SPU.td)

tablegen(SPUGenInstrNames.inc -gen-instr-enums)
tablegen(SPUGenRegisterNames.inc -gen-register-enums)
tablegen(SPUGenAsmWriter.inc -gen-asm-writer)
tablegen(SPUGenCodeEmitter.inc -gen-emitter)
tablegen(SPUGenRegisterInfo.h.inc -gen-register-desc-header)
tablegen(SPUGenRegisterInfo.inc -gen-register-desc)
tablegen(SPUGenInstrInfo.inc -gen-instr-desc)
tablegen(SPUGenDAGISel.inc -gen-dag-isel)
tablegen(SPUGenSubtarget.inc -gen-subtarget)
tablegen(SPUGenCallingConv.inc -gen-callingconv)

add_llvm_target(CellSPUCodeGen
  SPUAsmPrinter.cpp
  SPUFrameInfo.cpp
  SPUHazardRecognizers.cpp
  SPUInstrInfo.cpp
  SPUISelDAGToDAG.cpp
  SPUISelLowering.cpp
  SPUFrameInfo.cpp
  SPUMCAsmInfo.cpp
  SPURegisterInfo.cpp
  SPUSubtarget.cpp
  SPUTargetMachine.cpp
  SPUSelectionDAGInfo.cpp
  )
