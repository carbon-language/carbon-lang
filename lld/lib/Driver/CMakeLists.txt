set(LLVM_TARGET_DEFINITIONS UniversalDriverOptions.td)
tablegen(LLVM UniversalDriverOptions.inc -gen-opt-parser-defs)
set(LLVM_TARGET_DEFINITIONS GnuLdOptions.td)
tablegen(LLVM GnuLdOptions.inc -gen-opt-parser-defs)
set(LLVM_TARGET_DEFINITIONS CoreOptions.td)
tablegen(LLVM CoreOptions.inc -gen-opt-parser-defs)
set(LLVM_TARGET_DEFINITIONS DarwinLdOptions.td)
tablegen(LLVM DarwinLdOptions.inc -gen-opt-parser-defs)
set(LLVM_TARGET_DEFINITIONS WinLinkOptions.td)
tablegen(LLVM WinLinkOptions.inc -gen-opt-parser-defs)
add_public_tablegen_target(DriverOptionsTableGen)

add_llvm_library(lldDriver
  CoreDriver.cpp
  DarwinLdDriver.cpp
  Driver.cpp
  GnuLdDriver.cpp
  UniversalDriver.cpp
  WinLinkDriver.cpp
  WinLinkModuleDef.cpp
  LINK_LIBS
    lldConfig
    lldMachO
    lldCOFF
    lldPECOFF
    lldELF
    lldAArch64ELFTarget
    lldARMELFTarget
    lldHexagonELFTarget
    lldMipsELFTarget
    lldX86ELFTarget
    lldExampleSubTarget
    lldX86_64ELFTarget
    lldCore
    lldReaderWriter
    lldYAML
    LLVMObject
    LLVMOption
    LLVMSupport
  )

add_dependencies(lldDriver DriverOptionsTableGen)

