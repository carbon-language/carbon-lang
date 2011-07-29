set(LLVM_TARGET_DEFINITIONS Blackfin.td)

tablegen(BlackfinGenRegisterInfo.inc -gen-register-info)
tablegen(BlackfinGenInstrInfo.inc -gen-instr-info)
tablegen(BlackfinGenAsmWriter.inc -gen-asm-writer)
tablegen(BlackfinGenDAGISel.inc -gen-dag-isel)
tablegen(BlackfinGenSubtargetInfo.inc -gen-subtarget)
tablegen(BlackfinGenCallingConv.inc -gen-callingconv)
tablegen(BlackfinGenIntrinsics.inc -gen-tgt-intrinsic)
add_public_tablegen_target(BlackfinCommonTableGen)

add_llvm_target(BlackfinCodeGen
  BlackfinAsmPrinter.cpp
  BlackfinInstrInfo.cpp
  BlackfinIntrinsicInfo.cpp
  BlackfinISelDAGToDAG.cpp
  BlackfinISelLowering.cpp
  BlackfinFrameLowering.cpp
  BlackfinRegisterInfo.cpp
  BlackfinSubtarget.cpp
  BlackfinTargetMachine.cpp
  BlackfinSelectionDAGInfo.cpp
  )

add_llvm_library_dependencies(LLVMBlackfinCodeGen
  LLVMAsmPrinter
  LLVMBlackfinDesc
  LLVMBlackfinInfo
  LLVMCodeGen
  LLVMCore
  LLVMMC
  LLVMSelectionDAG
  LLVMSupport
  LLVMTarget
  )

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
