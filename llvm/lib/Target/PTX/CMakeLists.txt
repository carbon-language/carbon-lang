set(LLVM_TARGET_DEFINITIONS PTX.td)

tablegen(PTXGenAsmWriter.inc -gen-asm-writer)
tablegen(PTXGenDAGISel.inc -gen-dag-isel)
tablegen(PTXGenInstrInfo.inc -gen-instr-info)
tablegen(PTXGenRegisterInfo.inc -gen-register-info)
tablegen(PTXGenSubtargetInfo.inc -gen-subtarget)
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
  PTXParamManager.cpp
  PTXRegAlloc.cpp
  PTXRegisterInfo.cpp
  PTXSelectionDAGInfo.cpp
  PTXSubtarget.cpp
  PTXTargetMachine.cpp
  )

add_llvm_library_dependencies(LLVMPTXCodeGen
  LLVMAnalysis
  LLVMAsmPrinter
  LLVMCodeGen
  LLVMCore
  LLVMMC
  LLVMPTXDesc
  LLVMPTXInfo
  LLVMSelectionDAG
  LLVMSupport
  LLVMTarget
  )

add_subdirectory(TargetInfo)
add_subdirectory(InstPrinter)
add_subdirectory(MCTargetDesc)

