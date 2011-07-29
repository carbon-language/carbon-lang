set(LLVM_TARGET_DEFINITIONS SPU.td)

tablegen(SPUGenAsmWriter.inc -gen-asm-writer)
tablegen(SPUGenCodeEmitter.inc -gen-emitter)
tablegen(SPUGenRegisterInfo.inc -gen-register-info)
tablegen(SPUGenInstrInfo.inc -gen-instr-info)
tablegen(SPUGenDAGISel.inc -gen-dag-isel)
tablegen(SPUGenSubtargetInfo.inc -gen-subtarget)
tablegen(SPUGenCallingConv.inc -gen-callingconv)
add_public_tablegen_target(CellSPUCommonTableGen)

add_llvm_target(CellSPUCodeGen
  SPUAsmPrinter.cpp
  SPUHazardRecognizers.cpp
  SPUInstrInfo.cpp
  SPUISelDAGToDAG.cpp
  SPUISelLowering.cpp
  SPUFrameLowering.cpp
  SPURegisterInfo.cpp
  SPUSubtarget.cpp
  SPUTargetMachine.cpp
  SPUSelectionDAGInfo.cpp
  SPUNopFiller.cpp
  )

add_llvm_library_dependencies(LLVMCellSPUCodeGen
  LLVMAsmPrinter
  LLVMCellSPUDesc
  LLVMCellSPUInfo
  LLVMCodeGen
  LLVMCore
  LLVMMC
  LLVMSelectionDAG
  LLVMSupport
  LLVMTarget
  )

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
