set(LLVM_TARGET_DEFINITIONS XCore.td)

llvm_tablegen(XCoreGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(XCoreGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(XCoreGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(XCoreGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(XCoreGenCallingConv.inc -gen-callingconv)
llvm_tablegen(XCoreGenSubtargetInfo.inc -gen-subtarget)
add_public_tablegen_target(XCoreCommonTableGen)

add_llvm_target(XCoreCodeGen
  XCoreAsmPrinter.cpp
  XCoreFrameLowering.cpp
  XCoreInstrInfo.cpp
  XCoreISelDAGToDAG.cpp
  XCoreISelLowering.cpp
  XCoreRegisterInfo.cpp
  XCoreSubtarget.cpp
  XCoreTargetMachine.cpp
  XCoreTargetObjectFile.cpp
  XCoreSelectionDAGInfo.cpp
  )

add_llvm_library_dependencies(LLVMXCoreCodeGen
  LLVMAsmPrinter
  LLVMCodeGen
  LLVMCore
  LLVMMC
  LLVMSelectionDAG
  LLVMSupport
  LLVMTarget
  LLVMXCoreDesc
  LLVMXCoreInfo
  )

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
