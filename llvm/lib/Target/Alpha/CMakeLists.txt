set(LLVM_TARGET_DEFINITIONS Alpha.td)

tablegen(AlphaGenRegisterInfo.inc -gen-register-info)
tablegen(AlphaGenInstrInfo.inc -gen-instr-info)
tablegen(AlphaGenAsmWriter.inc -gen-asm-writer)
tablegen(AlphaGenDAGISel.inc -gen-dag-isel)
tablegen(AlphaGenCallingConv.inc -gen-callingconv)
tablegen(AlphaGenSubtargetInfo.inc -gen-subtarget)

add_llvm_target(AlphaCodeGen
  AlphaAsmPrinter.cpp
  AlphaBranchSelector.cpp
  AlphaInstrInfo.cpp
  AlphaISelDAGToDAG.cpp
  AlphaISelLowering.cpp
  AlphaFrameLowering.cpp
  AlphaLLRP.cpp
  AlphaMCAsmInfo.cpp
  AlphaRegisterInfo.cpp
  AlphaSubtarget.cpp
  AlphaTargetMachine.cpp
  AlphaSelectionDAGInfo.cpp
  )

add_subdirectory(TargetInfo)
