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
  InputGraph.cpp
  CoreDriver.cpp
  DarwinLdDriver.cpp
  Driver.cpp
  GnuLdDriver.cpp
  WinLinkDriver.cpp
  UniversalDriver.cpp
  )

add_dependencies(lldDriver DriverOptionsTableGen)

target_link_libraries(lldDriver
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
