set(LLVM_TARGET_DEFINITIONS MBlaze.td)

llvm_tablegen(MBlazeGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(MBlazeGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(MBlazeGenCodeEmitter.inc -gen-emitter)
llvm_tablegen(MBlazeGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(MBlazeGenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(MBlazeGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(MBlazeGenCallingConv.inc -gen-callingconv)
llvm_tablegen(MBlazeGenSubtargetInfo.inc -gen-subtarget)
llvm_tablegen(MBlazeGenIntrinsics.inc -gen-tgt-intrinsic)
llvm_tablegen(MBlazeGenEDInfo.inc -gen-enhanced-disassembly-info)
add_public_tablegen_target(MBlazeCommonTableGen)

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
  MBlazeMCInstLower.cpp
  MBlazeELFWriterInfo.cpp
  )

add_llvm_library_dependencies(LLVMMBlazeCodeGen
  LLVMAsmPrinter
  LLVMCodeGen
  LLVMCore
  LLVMMBlazeAsmPrinter
  LLVMMBlazeDesc
  LLVMMBlazeInfo
  LLVMMC
  LLVMSelectionDAG
  LLVMSupport
  LLVMTarget
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(InstPrinter)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
