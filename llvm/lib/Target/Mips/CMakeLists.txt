set(LLVM_TARGET_DEFINITIONS Mips.td)

tablegen(MipsGenRegisterInfo.inc -gen-register-info)
tablegen(MipsGenInstrInfo.inc -gen-instr-info)
tablegen(MipsGenAsmWriter.inc -gen-asm-writer)
tablegen(MipsGenDAGISel.inc -gen-dag-isel)
tablegen(MipsGenCallingConv.inc -gen-callingconv)
tablegen(MipsGenSubtargetInfo.inc -gen-subtarget)

add_llvm_target(MipsCodeGen
  MipsAsmPrinter.cpp
  MipsCodeEmitter.cpp
  MipsDelaySlotFiller.cpp
  MipsEmitGPRestore.cpp
  MipsExpandPseudo.cpp
  MipsJITInfo.cpp
  MipsInstrInfo.cpp
  MipsISelDAGToDAG.cpp
  MipsISelLowering.cpp
  MipsFrameLowering.cpp
  MipsMCInstLower.cpp
  MipsMCSymbolRefExpr.cpp
  MipsRegisterInfo.cpp
  MipsSubtarget.cpp
  MipsTargetMachine.cpp
  MipsTargetObjectFile.cpp
  MipsSelectionDAGInfo.cpp
  )

add_subdirectory(InstPrinter)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
