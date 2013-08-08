add_llvm_library(LLVMAsmPrinter
  ARMException.cpp
  AsmPrinter.cpp
  AsmPrinterDwarf.cpp
  AsmPrinterInlineAsm.cpp
  DIE.cpp
  DIEHash.cpp
  DwarfAccelTable.cpp
  DwarfCFIException.cpp
  DwarfCompileUnit.cpp
  DwarfDebug.cpp
  DwarfException.cpp
  ErlangGCPrinter.cpp
  OcamlGCPrinter.cpp
  Win64Exception.cpp
  )

add_dependencies(LLVMAsmPrinter intrinsics_gen)
