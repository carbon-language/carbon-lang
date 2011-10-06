set(LLVM_TARGET_DEFINITIONS MSP430.td)

llvm_tablegen(MSP430GenRegisterInfo.inc -gen-register-info)
llvm_tablegen(MSP430GenInstrInfo.inc -gen-instr-info)
llvm_tablegen(MSP430GenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(MSP430GenDAGISel.inc -gen-dag-isel)
llvm_tablegen(MSP430GenCallingConv.inc -gen-callingconv)
llvm_tablegen(MSP430GenSubtargetInfo.inc -gen-subtarget)
add_public_tablegen_target(MSP430CommonTableGen)

add_llvm_target(MSP430CodeGen
  MSP430BranchSelector.cpp
  MSP430ISelDAGToDAG.cpp
  MSP430ISelLowering.cpp
  MSP430InstrInfo.cpp
  MSP430FrameLowering.cpp
  MSP430RegisterInfo.cpp
  MSP430Subtarget.cpp
  MSP430TargetMachine.cpp
  MSP430SelectionDAGInfo.cpp
  MSP430AsmPrinter.cpp
  MSP430MCInstLower.cpp
  )

add_llvm_library_dependencies(LLVMMSP430CodeGen
  LLVMAsmPrinter
  LLVMCodeGen
  LLVMCore
  LLVMMC
  LLVMMSP430AsmPrinter
  LLVMMSP430Desc
  LLVMMSP430Info
  LLVMSelectionDAG
  LLVMSupport
  LLVMTarget
  )

add_subdirectory(InstPrinter)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
