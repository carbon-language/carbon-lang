add_llvm_library(LLVMAsmPrinter
  AddressPool.cpp
  ARMException.cpp
  AsmPrinter.cpp
  AsmPrinterDwarf.cpp
  AsmPrinterInlineAsm.cpp
  DIE.cpp
  DIEHash.cpp
  DwarfAccelTable.cpp
  DwarfCFIException.cpp
  DwarfDebug.cpp
  DwarfException.cpp
  DwarfFile.cpp
  DwarfStringPool.cpp
  DwarfUnit.cpp
  ErlangGCPrinter.cpp
  OcamlGCPrinter.cpp
  Win64Exception.cpp
  WinCodeViewLineTables.cpp
  )

add_dependencies(LLVMAsmPrinter intrinsics_gen)
