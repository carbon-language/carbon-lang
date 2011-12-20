set(LLVM_TARGET_DEFINITIONS PTX.td)

tablegen(LLVM PTXGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM PTXGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM PTXGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM PTXGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM PTXGenSubtargetInfo.inc -gen-subtarget)
add_public_tablegen_target(PTXCommonTableGen)

add_llvm_target(PTXCodeGen
  PTXAsmPrinter.cpp
  PTXISelDAGToDAG.cpp
  PTXISelLowering.cpp
  PTXInstrInfo.cpp
  PTXFPRoundingModePass.cpp
  PTXFrameLowering.cpp
  PTXMCAsmStreamer.cpp
  PTXMCInstLower.cpp
  PTXMFInfoExtract.cpp
  PTXMachineFunctionInfo.cpp
  PTXParamManager.cpp
  PTXRegAlloc.cpp
  PTXRegisterInfo.cpp
  PTXSelectionDAGInfo.cpp
  PTXSubtarget.cpp
  PTXTargetMachine.cpp
  )

add_subdirectory(TargetInfo)
add_subdirectory(InstPrinter)
add_subdirectory(MCTargetDesc)

