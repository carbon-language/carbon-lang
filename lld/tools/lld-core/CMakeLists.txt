set(LLVM_USED_LIBS
  lldPasses
  lldMachO
  lldPECOFF
  lldELF
  lldNative
  lldYAML
  lldReaderWriter
  lldCore
 )

set(LLVM_LINK_COMPONENTS
  support
  Object
  )

add_lld_executable(lld-core
  lld-core.cpp
  )

install(TARGETS lld-core
  RUNTIME DESTINATION bin)
