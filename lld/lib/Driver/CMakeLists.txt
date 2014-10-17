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

add_lld_library(lldDriver
  CoreDriver.cpp
  DarwinInputGraph.cpp
  DarwinLdDriver.cpp
  Driver.cpp
  GnuLdDriver.cpp
  GnuLdInputGraph.cpp
  UniversalDriver.cpp
  WinLinkDriver.cpp
  WinLinkInputGraph.cpp
  WinLinkModuleDef.cpp
  )

add_dependencies(lldDriver DriverOptionsTableGen)

target_link_libraries(lldDriver ${cmake_2_8_12_INTERFACE}
  lldConfig
  lldPasses
  lldMachO
  lldPECOFF
  lldELF
  lldCore
  lldNative
  lldReaderWriter
  lldYAML
  LLVMObject
  LLVMOption
  LLVMSupport
  )
