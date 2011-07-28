add_llvm_library(LLVMMSP430Desc
  MSP430MCTargetDesc.cpp
  MSP430MCAsmInfo.cpp
  )
add_dependencies(LLVMMSP430Desc MSP430CommonTableGen)

target_link_libraries(LLVMMSP430Desc LLVMMSP430AsmPrinter)
