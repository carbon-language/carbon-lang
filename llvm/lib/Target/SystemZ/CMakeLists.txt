set(LLVM_TARGET_DEFINITIONS SystemZ.td)

tablegen(SystemZGenRegisterInfo.inc -gen-register-info)
tablegen(SystemZGenInstrInfo.inc -gen-instr-info)
tablegen(SystemZGenAsmWriter.inc -gen-asm-writer)
tablegen(SystemZGenDAGISel.inc -gen-dag-isel)
tablegen(SystemZGenCallingConv.inc -gen-callingconv)
tablegen(SystemZGenSubtargetInfo.inc -gen-subtarget)

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
