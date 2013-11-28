add_llvm_library(LLVMMSP430Info
  MSP430TargetInfo.cpp
  )

add_dependencies(LLVMMSP430Info MSP430CommonTableGen)
