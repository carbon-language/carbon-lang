set(LLVM_TARGET_DEFINITIONS ARM.td)

tablegen(ARMGenRegisterInfo.h.inc -gen-register-desc-header)
tablegen(ARMGenRegisterNames.inc -gen-register-enums)
tablegen(ARMGenRegisterInfo.inc -gen-register-desc)
tablegen(ARMGenInstrNames.inc -gen-instr-enums)
tablegen(ARMGenInstrInfo.inc -gen-instr-desc)
tablegen(ARMGenCodeEmitter.inc -gen-emitter)
tablegen(ARMGenAsmWriter.inc -gen-asm-writer)
tablegen(ARMGenDAGISel.inc -gen-dag-isel)
tablegen(ARMGenCallingConv.inc -gen-callingconv)
tablegen(ARMGenSubtarget.inc -gen-subtarget)

add_llvm_target(ARMCodeGen
  ARMBaseInstrInfo.cpp
  ARMBaseRegisterInfo.cpp
  ARMCodeEmitter.cpp
  ARMConstantIslandPass.cpp
  ARMConstantPoolValue.cpp
  ARMInstrInfo.cpp
  ARMISelDAGToDAG.cpp
  ARMISelLowering.cpp
  ARMJITInfo.cpp
  ARMLoadStoreOptimizer.cpp
  ARMRegisterInfo.cpp
  ARMSubtarget.cpp
  ARMMCAsmInfo.cpp
  ARMTargetMachine.cpp
  NEONPreAllocPass.cpp
  Thumb1InstrInfo.cpp
  Thumb1RegisterInfo.cpp
  Thumb2ITBlockPass.cpp
  Thumb2InstrInfo.cpp
  Thumb2RegisterInfo.cpp
  Thumb2SizeReduction.cpp
  )

target_link_libraries (LLVMARMCodeGen LLVMSelectionDAG)
