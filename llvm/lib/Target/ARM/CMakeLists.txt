set(LLVM_TARGET_DEFINITIONS ARM.td)

llvm_tablegen(ARMGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(ARMGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(ARMGenCodeEmitter.inc -gen-emitter)
llvm_tablegen(ARMGenMCCodeEmitter.inc -gen-emitter -mc-emitter)
llvm_tablegen(ARMGenMCPseudoLowering.inc -gen-pseudo-lowering)
llvm_tablegen(ARMGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(ARMGenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(ARMGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(ARMGenFastISel.inc -gen-fast-isel)
llvm_tablegen(ARMGenCallingConv.inc -gen-callingconv)
llvm_tablegen(ARMGenSubtargetInfo.inc -gen-subtarget)
llvm_tablegen(ARMGenEDInfo.inc -gen-enhanced-disassembly-info)
llvm_tablegen(ARMGenDisassemblerTables.inc -gen-disassembler)
add_public_tablegen_target(ARMCommonTableGen)

add_llvm_target(ARMCodeGen
  ARMAsmPrinter.cpp
  ARMBaseInstrInfo.cpp
  ARMBaseRegisterInfo.cpp
  ARMCodeEmitter.cpp
  ARMConstantIslandPass.cpp
  ARMConstantPoolValue.cpp
  ARMELFWriterInfo.cpp
  ARMExpandPseudoInsts.cpp
  ARMFastISel.cpp
  ARMFrameLowering.cpp
  ARMGlobalMerge.cpp
  ARMHazardRecognizer.cpp
  ARMISelDAGToDAG.cpp
  ARMISelLowering.cpp
  ARMInstrInfo.cpp
  ARMJITInfo.cpp
  ARMLoadStoreOptimizer.cpp
  ARMMCInstLower.cpp
  ARMRegisterInfo.cpp
  ARMSelectionDAGInfo.cpp
  ARMSubtarget.cpp
  ARMTargetMachine.cpp
  ARMTargetObjectFile.cpp
  MLxExpansionPass.cpp
  Thumb1FrameLowering.cpp
  Thumb1InstrInfo.cpp
  Thumb1RegisterInfo.cpp
  Thumb2ITBlockPass.cpp
  Thumb2InstrInfo.cpp
  Thumb2RegisterInfo.cpp
  Thumb2SizeReduction.cpp
  )

add_llvm_library_dependencies(LLVMARMCodeGen
  LLVMARMAsmPrinter
  LLVMARMDesc
  LLVMARMInfo
  LLVMAnalysis
  LLVMAsmPrinter
  LLVMCodeGen
  LLVMCore
  LLVMMC
  LLVMSelectionDAG
  LLVMSupport
  LLVMTarget
  )

# workaround for hanging compilation on MSVC10
if( MSVC_VERSION EQUAL 1600 )
set_property(
  SOURCE ARMISelLowering.cpp
  PROPERTY COMPILE_FLAGS "/Od"
  )
endif()

add_subdirectory(TargetInfo)
add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(InstPrinter)
add_subdirectory(MCTargetDesc)
