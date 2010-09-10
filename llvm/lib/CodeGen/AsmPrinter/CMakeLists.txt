add_llvm_library(LLVMAsmPrinter
  AsmPrinter.cpp
  AsmPrinterDwarf.cpp
  AsmPrinterInlineAsm.cpp
  DIE.cpp
  DwarfDebug.cpp
  DwarfException.cpp
  OcamlGCPrinter.cpp
  )

target_link_libraries (LLVMAsmPrinter
  LLVMAnalysis
  LLVMCodeGen
  LLVMCore
  LLVMMC
  LLVMMCParser
  LLVMTarget
  )
