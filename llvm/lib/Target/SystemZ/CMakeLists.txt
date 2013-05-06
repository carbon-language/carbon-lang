set(LLVM_TARGET_DEFINITIONS SystemZ.td)

tablegen(LLVM SystemZGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM SystemZGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM SystemZGenCallingConv.inc -gen-callingconv)
tablegen(LLVM SystemZGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM SystemZGenMCCodeEmitter.inc -gen-emitter -mc-emitter)
tablegen(LLVM SystemZGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM SystemZGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM SystemZGenSubtargetInfo.inc -gen-subtarget)
add_public_tablegen_target(SystemZCommonTableGen)

add_llvm_target(SystemZCodeGen
  SystemZAsmPrinter.cpp
  SystemZCallingConv.cpp
  SystemZConstantPoolValue.cpp
  SystemZFrameLowering.cpp
  SystemZISelDAGToDAG.cpp
  SystemZISelLowering.cpp
  SystemZInstrInfo.cpp
  SystemZMCInstLower.cpp
  SystemZRegisterInfo.cpp
  SystemZSubtarget.cpp
  SystemZTargetMachine.cpp
  )

add_dependencies(LLVMSystemZCodeGen intrinsics_gen)

add_subdirectory(AsmParser)
add_subdirectory(InstPrinter)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
