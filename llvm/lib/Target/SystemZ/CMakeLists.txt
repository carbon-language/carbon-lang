set(LLVM_TARGET_DEFINITIONS SystemZ.td)

tablegen(SystemZGenRegisterInfo.h.inc -gen-register-desc-header)
tablegen(SystemZGenRegisterNames.inc -gen-register-enums)
tablegen(SystemZGenRegisterInfo.inc -gen-register-desc)
tablegen(SystemZGenInstrNames.inc -gen-instr-enums)
tablegen(SystemZGenInstrInfo.inc -gen-instr-desc)
tablegen(SystemZGenAsmWriter.inc -gen-asm-writer)
tablegen(SystemZGenDAGISel.inc -gen-dag-isel)
tablegen(SystemZGenCallingConv.inc -gen-callingconv)
tablegen(SystemZGenSubtarget.inc -gen-subtarget)

add_llvm_target(SystemZCodeGen
  SystemZISelDAGToDAG.cpp
  SystemZISelLowering.cpp
  SystemZInstrInfo.cpp
  SystemZRegisterInfo.cpp
  SystemZSubtarget.cpp
  SystemZMCAsmInfo.cpp
  SystemZTargetMachine.cpp
  )

target_link_libraries (LLVMSystemZCodeGen LLVMSelectionDAG)
