set(LLVM_TARGET_DEFINITIONS MBlaze.td)

tablegen(MBlazeGenRegisterInfo.inc -gen-register-info)
tablegen(MBlazeGenInstrInfo.inc -gen-instr-info)
tablegen(MBlazeGenCodeEmitter.inc -gen-emitter)
tablegen(MBlazeGenAsmWriter.inc -gen-asm-writer)
tablegen(MBlazeGenAsmMatcher.inc -gen-asm-matcher)
tablegen(MBlazeGenDAGISel.inc -gen-dag-isel)
tablegen(MBlazeGenCallingConv.inc -gen-callingconv)
tablegen(MBlazeGenSubtargetInfo.inc -gen-subtarget)
tablegen(MBlazeGenIntrinsics.inc -gen-tgt-intrinsic)
tablegen(MBlazeGenEDInfo.inc -gen-enhanced-disassembly-info)

add_llvm_target(MBlazeCodeGen
  MBlazeDelaySlotFiller.cpp
  MBlazeInstrInfo.cpp
  MBlazeISelDAGToDAG.cpp
  MBlazeISelLowering.cpp
  MBlazeFrameLowering.cpp
  MBlazeRegisterInfo.cpp
  MBlazeSubtarget.cpp
  MBlazeTargetMachine.cpp
  MBlazeTargetObjectFile.cpp
  MBlazeIntrinsicInfo.cpp
  MBlazeSelectionDAGInfo.cpp
  MBlazeAsmPrinter.cpp
  MBlazeAsmBackend.cpp
  MBlazeMCInstLower.cpp
  MBlazeELFWriterInfo.cpp
  MBlazeMCCodeEmitter.cpp
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(InstPrinter)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
