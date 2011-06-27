set(LLVM_TARGET_DEFINITIONS Sparc.td)

tablegen(SparcGenRegisterInfo.inc -gen-register-info)
tablegen(SparcGenInstrNames.inc -gen-instr-enums)
tablegen(SparcGenInstrInfo.inc -gen-instr-desc)
tablegen(SparcGenAsmWriter.inc -gen-asm-writer)
tablegen(SparcGenDAGISel.inc -gen-dag-isel)
tablegen(SparcGenSubtarget.inc -gen-subtarget)
tablegen(SparcGenCallingConv.inc -gen-callingconv)

add_llvm_target(SparcCodeGen
  DelaySlotFiller.cpp
  FPMover.cpp
  SparcAsmPrinter.cpp
  SparcInstrInfo.cpp
  SparcISelDAGToDAG.cpp
  SparcISelLowering.cpp
  SparcFrameLowering.cpp
  SparcMCAsmInfo.cpp
  SparcRegisterInfo.cpp
  SparcSubtarget.cpp
  SparcTargetMachine.cpp
  SparcSelectionDAGInfo.cpp
  )

add_subdirectory(TargetInfo)
