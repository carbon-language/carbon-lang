set(LLVM_TARGET_DEFINITIONS XCore.td)

tablegen(XCoreGenRegisterDesc.inc -gen-register-desc)
tablegen(XCoreGenRegisterInfo.h.inc -gen-register-info-header)
tablegen(XCoreGenRegisterInfo.inc -gen-register-info)
tablegen(XCoreGenInstrNames.inc -gen-instr-enums)
tablegen(XCoreGenInstrInfo.inc -gen-instr-desc)
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
