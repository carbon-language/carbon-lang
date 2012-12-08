set(LLVM_TARGET_DEFINITIONS LDOptions.td)
tablegen(LLVM LDOptions.inc -gen-opt-parser-defs)
set(LLVM_TARGET_DEFINITIONS CoreOptions.td)
tablegen(LLVM CoreOptions.inc -gen-opt-parser-defs)
add_public_tablegen_target(DriverOptionsTableGen)

add_lld_library(lldDriver
  Driver.cpp
  Drivers.cpp
  LinkerInvocation.cpp
  Target.cpp
  Targets.cpp
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
