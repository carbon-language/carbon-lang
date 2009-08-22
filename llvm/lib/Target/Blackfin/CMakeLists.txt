set(LLVM_TARGET_DEFINITIONS Blackfin.td)

tablegen(BlackfinGenRegisterInfo.h.inc -gen-register-desc-header)
tablegen(BlackfinGenRegisterNames.inc -gen-register-enums)
tablegen(BlackfinGenRegisterInfo.inc -gen-register-desc)
tablegen(BlackfinGenInstrNames.inc -gen-instr-enums)
tablegen(BlackfinGenInstrInfo.inc -gen-instr-desc)
tablegen(BlackfinGenAsmWriter.inc -gen-asm-writer)
tablegen(BlackfinGenDAGISel.inc -gen-dag-isel)
tablegen(BlackfinGenSubtarget.inc -gen-subtarget)
tablegen(BlackfinGenCallingConv.inc -gen-callingconv)

add_llvm_target(BlackfinCodeGen
  BlackfinInstrInfo.cpp
  BlackfinISelDAGToDAG.cpp
  BlackfinISelLowering.cpp
  BlackfinRegisterInfo.cpp
  BlackfinSubtarget.cpp
  BlackfinMCAsmInfo.cpp
  BlackfinTargetMachine.cpp
  )
