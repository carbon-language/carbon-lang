set(LLVM_TARGET_DEFINITIONS ARM64.td)

tablegen(LLVM ARM64GenRegisterInfo.inc -gen-register-info)
tablegen(LLVM ARM64GenInstrInfo.inc -gen-instr-info)
tablegen(LLVM ARM64GenMCCodeEmitter.inc -gen-emitter -mc-emitter)
tablegen(LLVM ARM64GenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM ARM64GenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM ARM64GenAsmWriter1.inc -gen-asm-writer -asmwriternum=1)
tablegen(LLVM ARM64GenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM ARM64GenDAGISel.inc -gen-dag-isel)
tablegen(LLVM ARM64GenFastISel.inc -gen-fast-isel)
tablegen(LLVM ARM64GenCallingConv.inc -gen-callingconv)
tablegen(LLVM ARM64GenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM ARM64GenDisassemblerTables.inc -gen-disassembler)
add_public_tablegen_target(ARM64CommonTableGen)

add_llvm_target(ARM64CodeGen
  ARM64AddressTypePromotion.cpp
  ARM64AdvSIMDScalarPass.cpp
  ARM64AsmPrinter.cpp
  ARM64BranchRelaxation.cpp
  ARM64CleanupLocalDynamicTLSPass.cpp
  ARM64CollectLOH.cpp
  ARM64ConditionalCompares.cpp
  ARM64DeadRegisterDefinitionsPass.cpp
  ARM64ExpandPseudoInsts.cpp
  ARM64FastISel.cpp
  ARM64FrameLowering.cpp
  ARM64ISelDAGToDAG.cpp
  ARM64ISelLowering.cpp
  ARM64InstrInfo.cpp
  ARM64LoadStoreOptimizer.cpp
  ARM64MCInstLower.cpp
  ARM64PromoteConstant.cpp
  ARM64RegisterInfo.cpp
  ARM64SelectionDAGInfo.cpp
  ARM64StorePairSuppress.cpp
  ARM64Subtarget.cpp
  ARM64TargetMachine.cpp
  ARM64TargetObjectFile.cpp
  ARM64TargetTransformInfo.cpp
)

add_dependencies(LLVMARM64CodeGen intrinsics_gen)

add_subdirectory(TargetInfo)
add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(InstPrinter)
add_subdirectory(MCTargetDesc)
