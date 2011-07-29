set(LLVM_TARGET_DEFINITIONS SystemZ.td)

tablegen(SystemZGenRegisterInfo.inc -gen-register-info)
tablegen(SystemZGenInstrInfo.inc -gen-instr-info)
tablegen(SystemZGenAsmWriter.inc -gen-asm-writer)
tablegen(SystemZGenDAGISel.inc -gen-dag-isel)
tablegen(SystemZGenCallingConv.inc -gen-callingconv)
tablegen(SystemZGenSubtargetInfo.inc -gen-subtarget)
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
