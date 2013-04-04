set(LLVM_TARGET_DEFINITIONS LDOptions.td)
tablegen(LLVM LDOptions.inc -gen-opt-parser-defs)
set(LLVM_TARGET_DEFINITIONS CoreOptions.td)
tablegen(LLVM CoreOptions.inc -gen-opt-parser-defs)
set(LLVM_TARGET_DEFINITIONS DarwinOptions.td)
tablegen(LLVM DarwinOptions.inc -gen-opt-parser-defs)
add_public_tablegen_target(DriverOptionsTableGen)

add_lld_library(lldDriver
  CoreDriver.cpp
  DarwinLdDriver.cpp
  Driver.cpp
  GnuLdDriver.cpp
  UniversalDriver.cpp
  )

add_dependencies(lldDriver DriverOptionsTableGen)

target_link_libraries(lldDriver
  lldPasses
  lldMachO
  lldPECOFF
  lldELF
  lldNative
  lldReaderWriter
  lldYAML
  lldCore
  LLVMObject
  LLVMOption
  LLVMSupport
  )
