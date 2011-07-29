set(LLVM_TARGET_DEFINITIONS Sparc.td)

tablegen(SparcGenRegisterInfo.inc -gen-register-info)
tablegen(SparcGenInstrInfo.inc -gen-instr-info)
tablegen(SparcGenAsmWriter.inc -gen-asm-writer)
tablegen(SparcGenDAGISel.inc -gen-dag-isel)
tablegen(SparcGenSubtargetInfo.inc -gen-subtarget)
tablegen(SparcGenCallingConv.inc -gen-callingconv)
add_public_tablegen_target(SparcCommonTableGen)

add_llvm_target(SparcCodeGen
  DelaySlotFiller.cpp
  FPMover.cpp
  SparcAsmPrinter.cpp
  SparcInstrInfo.cpp
  SparcISelDAGToDAG.cpp
  SparcISelLowering.cpp
  SparcFrameLowering.cpp
  SparcRegisterInfo.cpp
  SparcSubtarget.cpp
  SparcTargetMachine.cpp
  SparcSelectionDAGInfo.cpp
  )

add_llvm_library_dependencies(LLVMSparcCodeGen
  LLVMAsmPrinter
  LLVMCodeGen
  LLVMCore
  LLVMMC
  LLVMSelectionDAG
  LLVMSparcDesc
  LLVMSparcInfo
  LLVMSupport
  LLVMTarget
  )

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
