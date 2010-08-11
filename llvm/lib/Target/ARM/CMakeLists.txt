set(LLVM_TARGET_DEFINITIONS ARM.td)

tablegen(ARMGenRegisterInfo.h.inc -gen-register-desc-header)
tablegen(ARMGenRegisterNames.inc -gen-register-enums)
tablegen(ARMGenRegisterInfo.inc -gen-register-desc)
tablegen(ARMGenInstrNames.inc -gen-instr-enums)
tablegen(ARMGenInstrInfo.inc -gen-instr-desc)
tablegen(ARMGenCodeEmitter.inc -gen-emitter)
tablegen(ARMGenAsmWriter.inc -gen-asm-writer)
tablegen(ARMGenAsmMatcher.inc -gen-asm-matcher)
tablegen(ARMGenDAGISel.inc -gen-dag-isel)
tablegen(ARMGenFastISel.inc -gen-fast-isel)
tablegen(ARMGenCallingConv.inc -gen-callingconv)
tablegen(ARMGenSubtarget.inc -gen-subtarget)
tablegen(ARMGenEDInfo.inc -gen-enhanced-disassembly-info)

add_llvm_target(ARMCodeGen
  ARMAsmPrinter.cpp
  ARMBaseInstrInfo.cpp
  ARMBaseRegisterInfo.cpp
  ARMCodeEmitter.cpp
  ARMConstantIslandPass.cpp
  ARMConstantPoolValue.cpp
  ARMExpandPseudoInsts.cpp
  ARMFastISel.cpp
  ARMGlobalMerge.cpp
  ARMISelDAGToDAG.cpp
  ARMISelLowering.cpp
  ARMInstrInfo.cpp
  ARMJITInfo.cpp
  ARMLoadStoreOptimizer.cpp
  ARMMCAsmInfo.cpp
  ARMMCInstLower.cpp
  ARMRegisterInfo.cpp
  ARMSelectionDAGInfo.cpp
  ARMSubtarget.cpp
  ARMTargetMachine.cpp
  ARMTargetObjectFile.cpp
  NEONMoveFix.cpp
  NEONPreAllocPass.cpp
  Thumb1InstrInfo.cpp
  Thumb1RegisterInfo.cpp
  Thumb2HazardRecognizer.cpp
  Thumb2ITBlockPass.cpp
  Thumb2InstrInfo.cpp
  Thumb2RegisterInfo.cpp
  Thumb2SizeReduction.cpp
  )

target_link_libraries (LLVMARMCodeGen LLVMARMAsmPrinter LLVMSelectionDAG)
