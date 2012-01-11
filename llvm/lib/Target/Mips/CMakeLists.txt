set(LLVM_TARGET_DEFINITIONS Mips.td)

tablegen(LLVM MipsGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM MipsGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM MipsGenCodeEmitter.inc -gen-emitter)
tablegen(LLVM MipsGenMCCodeEmitter.inc -gen-emitter -mc-emitter)
tablegen(LLVM MipsGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM MipsGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM MipsGenCallingConv.inc -gen-callingconv)
tablegen(LLVM MipsGenSubtargetInfo.inc -gen-subtarget)
add_public_tablegen_target(MipsCommonTableGen)

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
  MipsMachineFunction.cpp
  MipsRegisterInfo.cpp
  MipsSubtarget.cpp
  MipsTargetMachine.cpp
  MipsTargetObjectFile.cpp
  MipsSelectionDAGInfo.cpp
  )

add_subdirectory(InstPrinter)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
add_subdirectory(AsmParser)
