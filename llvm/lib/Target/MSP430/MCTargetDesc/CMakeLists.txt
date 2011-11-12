add_llvm_library(LLVMMSP430Desc
  MSP430MCTargetDesc.cpp
  MSP430MCAsmInfo.cpp
  )

add_llvm_library_dependencies(LLVMMSP430Desc
  LLVMMC
  LLVMMSP430AsmPrinter
  LLVMMSP430Info
  LLVMSupport
  LLVMTarget
  )

add_dependencies(LLVMMSP430Desc MSP430CommonTableGen)
