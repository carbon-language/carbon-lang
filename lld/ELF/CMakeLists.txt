set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(ELFOptionsTableGen)

add_llvm_library(lldELF2
  Chunks.cpp
  Driver.cpp
  DriverUtils.cpp
  Error.cpp
  InputFiles.cpp
  SymbolTable.cpp
  Symbols.cpp
  Writer.cpp

  LINK_COMPONENTS
  Object
  Option
  MC
  Support
  )

add_dependencies(lldELF2 ELFOptionsTableGen)
