add_lld_executable(linker-script-test
  linker-script-test.cpp
  )

target_link_libraries(linker-script-test
  LLVMSupport
  lldReaderWriter
  )
