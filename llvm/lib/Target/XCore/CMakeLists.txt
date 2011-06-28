set(LLVM_TARGET_DEFINITIONS XCore.td)

tablegen(XCoreGenRegisterInfo.inc -gen-register-info)
tablegen(XCoreGenInstrInfo.inc -gen-instr-info)
tablegen(XCoreGenAsmWriter.inc -gen-asm-writer)
tablegen(XCoreGenDAGISel.inc -gen-dag-isel)
tablegen(XCoreGenCallingConv.inc -gen-callingconv)
tablegen(XCoreGenSubtarget.inc -gen-subtarget)

add_llvm_target(XCoreCodeGen
  XCoreAsmPrinter.cpp
  XCoreFrameLowering.cpp
  XCoreInstrInfo.cpp
  XCoreISelDAGToDAG.cpp
  XCoreISelLowering.cpp
  XCoreMCAsmInfo.cpp
  XCoreRegisterInfo.cpp
  XCoreSubtarget.cpp
  XCoreTargetMachine.cpp
  XCoreTargetObjectFile.cpp
  XCoreSelectionDAGInfo.cpp
  )

add_subdirectory(TargetInfo)
