set(LLVM_TARGET_DEFINITIONS Mips.td)

tablegen(MipsGenRegisterInfo.h.inc -gen-register-desc-header)
tablegen(MipsGenRegisterNames.inc -gen-register-enums)
tablegen(MipsGenRegisterInfo.inc -gen-register-desc)
tablegen(MipsGenInstrNames.inc -gen-instr-enums)
tablegen(MipsGenInstrInfo.inc -gen-instr-desc)
tablegen(MipsGenAsmWriter.inc -gen-asm-writer)
tablegen(MipsGenDAGISel.inc -gen-dag-isel)
tablegen(MipsGenCallingConv.inc -gen-callingconv)
tablegen(MipsGenSubtarget.inc -gen-subtarget)

add_llvm_target(MipsCodeGen
  MipsDelaySlotFiller.cpp
  MipsInstrInfo.cpp
  MipsISelDAGToDAG.cpp
  MipsISelLowering.cpp
  MipsRegisterInfo.cpp
  MipsSubtarget.cpp
  MipsMCAsmInfo.cpp
  MipsTargetMachine.cpp
  MipsTargetObjectFile.cpp
  )

target_link_libraries (LLVMMipsCodeGen LLVMSelectionDAG)
