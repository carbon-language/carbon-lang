set(LLVM_TARGET_DEFINITIONS MSP430.td)

tablegen(MSP430GenRegisterInfo.h.inc -gen-register-desc-header)
tablegen(MSP430GenRegisterNames.inc -gen-register-enums)
tablegen(MSP430GenRegisterInfo.inc -gen-register-desc)
tablegen(MSP430GenInstrNames.inc -gen-instr-enums)
tablegen(MSP430GenInstrInfo.inc -gen-instr-desc)
tablegen(MSP430GenAsmWriter.inc -gen-asm-writer)
tablegen(MSP430GenDAGISel.inc -gen-dag-isel)
tablegen(MSP430GenCallingConv.inc -gen-callingconv)
tablegen(MSP430GenSubtarget.inc -gen-subtarget)

add_llvm_target(MSP430CodeGen
  MSP430ISelDAGToDAG.cpp
  MSP430RegisterInfo.cpp
  MSP430MCAsmInfo.cpp
  MSP430InstrInfo.cpp
  MSP430ISelLowering.cpp
  MSP430Subtarget.cpp
  MSP430TargetMachine.cpp
  )

target_link_libraries (LLVMMSP430CodeGen LLVMSelectionDAG)
