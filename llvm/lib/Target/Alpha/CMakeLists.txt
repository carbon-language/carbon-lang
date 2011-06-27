set(LLVM_TARGET_DEFINITIONS Alpha.td)

tablegen(AlphaGenRegisterInfo.inc -gen-register-info)
tablegen(AlphaGenInstrNames.inc -gen-instr-enums)
tablegen(AlphaGenInstrInfo.inc -gen-instr-desc)
tablegen(AlphaGenAsmWriter.inc -gen-asm-writer)
tablegen(AlphaGenDAGISel.inc -gen-dag-isel)
tablegen(AlphaGenCallingConv.inc -gen-callingconv)
tablegen(AlphaGenSubtarget.inc -gen-subtarget)

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
