
add_lld_executable(lld-core
  lld-core.cpp
  )

target_link_libraries(lld-core
  lldPasses
  lldMachO
  lldPECOFF
  lldELF
  lldNative
  lldReaderWriter
  lldYAML
  lldCore
  lldDriver
  LLVMObject
  LLVMSupport
 )

install(TARGETS lld-core
  RUNTIME DESTINATION bin)
