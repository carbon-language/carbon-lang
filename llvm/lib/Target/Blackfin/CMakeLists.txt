set(LLVM_TARGET_DEFINITIONS Blackfin.td)

llvm_tablegen(BlackfinGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(BlackfinGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(BlackfinGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(BlackfinGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(BlackfinGenSubtargetInfo.inc -gen-subtarget)
llvm_tablegen(BlackfinGenCallingConv.inc -gen-callingconv)
llvm_tablegen(BlackfinGenIntrinsics.inc -gen-tgt-intrinsic)
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
