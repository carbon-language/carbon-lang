set(LLVM_TARGET_DEFINITIONS Sparc.td)

llvm_tablegen(SparcGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(SparcGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(SparcGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(SparcGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(SparcGenSubtargetInfo.inc -gen-subtarget)
llvm_tablegen(SparcGenCallingConv.inc -gen-callingconv)
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
