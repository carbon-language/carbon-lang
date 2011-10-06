set(LLVM_TARGET_DEFINITIONS SPU.td)

llvm_tablegen(SPUGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(SPUGenCodeEmitter.inc -gen-emitter)
llvm_tablegen(SPUGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(SPUGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(SPUGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(SPUGenSubtargetInfo.inc -gen-subtarget)
llvm_tablegen(SPUGenCallingConv.inc -gen-callingconv)
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
