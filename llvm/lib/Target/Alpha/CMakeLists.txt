set(LLVM_TARGET_DEFINITIONS Alpha.td)

llvm_tablegen(AlphaGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(AlphaGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(AlphaGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(AlphaGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(AlphaGenCallingConv.inc -gen-callingconv)
llvm_tablegen(AlphaGenSubtargetInfo.inc -gen-subtarget)
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
