set(LLVM_TARGET_DEFINITIONS Alpha.td)

tablegen(AlphaGenRegisterInfo.inc -gen-register-info)
tablegen(AlphaGenInstrInfo.inc -gen-instr-info)
tablegen(AlphaGenAsmWriter.inc -gen-asm-writer)
tablegen(AlphaGenDAGISel.inc -gen-dag-isel)
tablegen(AlphaGenCallingConv.inc -gen-callingconv)
tablegen(AlphaGenSubtargetInfo.inc -gen-subtarget)
add_public_tablegen_target(AlphaCommonTableGen)

add_llvm_target(AlphaCodeGen
  AlphaAsmPrinter.cpp
  AlphaBranchSelector.cpp
  AlphaInstrInfo.cpp
  AlphaISelDAGToDAG.cpp
  AlphaISelLowering.cpp
  AlphaFrameLowering.cpp
  AlphaLLRP.cpp
  AlphaRegisterInfo.cpp
  AlphaSubtarget.cpp
  AlphaTargetMachine.cpp
  AlphaSelectionDAGInfo.cpp
  )

add_llvm_library_dependencies(LLVMAlphaCodeGen
  LLVMAlphaDesc
  LLVMAlphaInfo
  LLVMAsmPrinter
  LLVMCodeGen
  LLVMCore
  LLVMMC
  LLVMSelectionDAG
  LLVMSupport
  LLVMTarget
  )

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
