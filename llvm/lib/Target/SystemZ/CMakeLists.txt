set(LLVM_TARGET_DEFINITIONS SystemZ.td)

llvm_tablegen(SystemZGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(SystemZGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(SystemZGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(SystemZGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(SystemZGenCallingConv.inc -gen-callingconv)
llvm_tablegen(SystemZGenSubtargetInfo.inc -gen-subtarget)
add_public_tablegen_target(SystemZCommonTableGen)

add_llvm_target(SystemZCodeGen
  SystemZAsmPrinter.cpp
  SystemZISelDAGToDAG.cpp
  SystemZISelLowering.cpp
  SystemZInstrInfo.cpp
  SystemZFrameLowering.cpp
  SystemZRegisterInfo.cpp
  SystemZSubtarget.cpp
  SystemZTargetMachine.cpp
  SystemZSelectionDAGInfo.cpp
  )

add_llvm_library_dependencies(LLVMSystemZCodeGen
  LLVMAsmPrinter
  LLVMCodeGen
  LLVMCore
  LLVMMC
  LLVMSelectionDAG
  LLVMSupport
  LLVMSystemZDesc
  LLVMSystemZInfo
  LLVMTarget
  )

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
