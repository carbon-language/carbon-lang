set(LLVM_TARGET_DEFINITIONS Blackfin.td)

tablegen(BlackfinGenRegisterInfo.h.inc -gen-register-desc-header)
tablegen(BlackfinGenRegisterNames.inc -gen-register-enums)
tablegen(BlackfinGenRegisterInfo.inc -gen-register-desc)
tablegen(BlackfinGenInstrNames.inc -gen-instr-enums)
tablegen(BlackfinGenInstrInfo.inc -gen-instr-desc)
tablegen(BlackfinGenAsmWriter.inc -gen-asm-writer)
tablegen(BlackfinGenDAGISel.inc -gen-dag-isel)
tablegen(BlackfinGenSubtarget.inc -gen-subtarget)
tablegen(BlackfinGenCallingConv.inc -gen-callingconv)
tablegen(BlackfinGenIntrinsics.inc -gen-tgt-intrinsic)

add_llvm_target(BlackfinCodeGen
  BlackfinAsmPrinter.cpp
  BlackfinInstrInfo.cpp
  BlackfinIntrinsicInfo.cpp
  BlackfinISelDAGToDAG.cpp
  BlackfinISelLowering.cpp
  BlackfinFrameLowering.cpp
  BlackfinMCAsmInfo.cpp
  BlackfinRegisterInfo.cpp
  BlackfinSubtarget.cpp
  BlackfinTargetMachine.cpp
  BlackfinSelectionDAGInfo.cpp
  )

add_subdirectory(TargetInfo)
