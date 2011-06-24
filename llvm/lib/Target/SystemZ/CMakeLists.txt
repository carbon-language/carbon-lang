set(LLVM_TARGET_DEFINITIONS SystemZ.td)

tablegen(SystemZGenRegisterNames.inc -gen-register-enums)
tablegen(SystemZGenRegisterDesc.inc -gen-register-desc)
tablegen(SystemZGenRegisterInfo.h.inc -gen-register-info-header)
tablegen(SystemZGenRegisterInfo.inc -gen-register-info)
tablegen(SystemZGenInstrNames.inc -gen-instr-enums)
tablegen(SystemZGenInstrInfo.inc -gen-instr-desc)
tablegen(SystemZGenAsmWriter.inc -gen-asm-writer)
tablegen(SystemZGenDAGISel.inc -gen-dag-isel)
tablegen(SystemZGenCallingConv.inc -gen-callingconv)
tablegen(SystemZGenSubtarget.inc -gen-subtarget)

add_llvm_target(SystemZCodeGen
  SystemZAsmPrinter.cpp
  SystemZISelDAGToDAG.cpp
  SystemZISelLowering.cpp
  SystemZInstrInfo.cpp
  SystemZFrameLowering.cpp
  SystemZMCAsmInfo.cpp
  SystemZRegisterInfo.cpp
  SystemZSubtarget.cpp
  SystemZTargetMachine.cpp
  SystemZSelectionDAGInfo.cpp
  )

add_subdirectory(TargetInfo)
