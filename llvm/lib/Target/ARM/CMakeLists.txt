set(LLVM_TARGET_DEFINITIONS ARM.td)

tablegen(ARMGenRegisterInfo.h.inc -gen-register-desc-header)
tablegen(ARMGenRegisterNames.inc -gen-register-enums)
tablegen(ARMGenRegisterInfo.inc -gen-register-desc)
tablegen(ARMGenInstrNames.inc -gen-instr-enums)
tablegen(ARMGenInstrInfo.inc -gen-instr-desc)
tablegen(ARMGenCodeEmitter.inc -gen-emitter)
tablegen(ARMGenMCCodeEmitter.inc -gen-emitter -mc-emitter)
tablegen(ARMGenAsmWriter.inc -gen-asm-writer)
tablegen(ARMGenAsmMatcher.inc -gen-asm-matcher)
tablegen(ARMGenDAGISel.inc -gen-dag-isel)
tablegen(ARMGenFastISel.inc -gen-fast-isel)
tablegen(ARMGenCallingConv.inc -gen-callingconv)
tablegen(ARMGenSubtarget.inc -gen-subtarget)
tablegen(ARMGenEDInfo.inc -gen-enhanced-disassembly-info)
tablegen(ARMGenDecoderTables.inc -gen-arm-decoder)

add_llvm_target(ARMCodeGen
  ARMAsmBackend.cpp
  ARMAsmPrinter.cpp
  ARMBaseInstrInfo.cpp
  ARMBaseRegisterInfo.cpp
  ARMCodeEmitter.cpp
  ARMConstantIslandPass.cpp
  ARMConstantPoolValue.cpp
  ARMELFWriterInfo.cpp
  ARMExpandPseudoInsts.cpp
  ARMFastISel.cpp
  ARMFrameInfo.cpp
  ARMGlobalMerge.cpp
  ARMHazardRecognizer.cpp
  ARMISelDAGToDAG.cpp
  ARMISelLowering.cpp
  ARMInstrInfo.cpp
  ARMJITInfo.cpp
  ARMMCCodeEmitter.cpp
  ARMLoadStoreOptimizer.cpp
  ARMMCAsmInfo.cpp
  ARMMCInstLower.cpp
  ARMRegisterInfo.cpp
  ARMSelectionDAGInfo.cpp
  ARMSubtarget.cpp
  ARMTargetMachine.cpp
  ARMTargetObjectFile.cpp
  NEONMoveFix.cpp
  Thumb1InstrInfo.cpp
  Thumb1FrameInfo.cpp
  Thumb1RegisterInfo.cpp
  Thumb2ITBlockPass.cpp
  Thumb2InstrInfo.cpp
  Thumb2RegisterInfo.cpp
  Thumb2SizeReduction.cpp
  )
