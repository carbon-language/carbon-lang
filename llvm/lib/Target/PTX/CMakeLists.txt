set(LLVM_TARGET_DEFINITIONS PTX.td)

tablegen(PTXGenAsmWriter.inc -gen-asm-writer)
tablegen(PTXGenCallingConv.inc -gen-callingconv)
tablegen(PTXGenDAGISel.inc -gen-dag-isel)
tablegen(PTXGenInstrInfo.inc -gen-instr-desc)
tablegen(PTXGenInstrNames.inc -gen-instr-enums)
tablegen(PTXGenRegisterInfo.inc -gen-register-info)
tablegen(PTXGenSubtarget.inc -gen-subtarget)

add_llvm_target(PTXCodeGen
  PTXAsmPrinter.cpp
  PTXISelDAGToDAG.cpp
  PTXISelLowering.cpp
  PTXInstrInfo.cpp
  PTXFrameLowering.cpp
  PTXMCAsmInfo.cpp
  PTXMCAsmStreamer.cpp
  PTXMFInfoExtract.cpp
  PTXRegisterInfo.cpp
  PTXSubtarget.cpp
  PTXTargetMachine.cpp
  )

add_subdirectory(TargetInfo)
