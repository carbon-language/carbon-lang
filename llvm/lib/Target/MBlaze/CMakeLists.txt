set(LLVM_TARGET_DEFINITIONS MBlaze.td)

tablegen(MBlazeGenRegisterInfo.h.inc -gen-register-desc-header)
tablegen(MBlazeGenRegisterNames.inc -gen-register-enums)
tablegen(MBlazeGenRegisterInfo.inc -gen-register-desc)
tablegen(MBlazeGenInstrNames.inc -gen-instr-enums)
tablegen(MBlazeGenInstrInfo.inc -gen-instr-desc)
tablegen(MBlazeGenAsmWriter.inc -gen-asm-writer)
tablegen(MBlazeGenDAGISel.inc -gen-dag-isel)
tablegen(MBlazeGenCallingConv.inc -gen-callingconv)
tablegen(MBlazeGenSubtarget.inc -gen-subtarget)
tablegen(MBlazeGenIntrinsics.inc -gen-tgt-intrinsic)

add_llvm_target(MBlazeCodeGen
  MBlazeDelaySlotFiller.cpp
  MBlazeInstrInfo.cpp
  MBlazeISelDAGToDAG.cpp
  MBlazeISelLowering.cpp
  MBlazeMCAsmInfo.cpp
  MBlazeRegisterInfo.cpp
  MBlazeSubtarget.cpp
  MBlazeTargetMachine.cpp
  MBlazeTargetObjectFile.cpp
  MBlazeIntrinsicInfo.cpp
  )

target_link_libraries (LLVMMBlazeCodeGen LLVMSelectionDAG)
