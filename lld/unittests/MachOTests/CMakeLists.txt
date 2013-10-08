
add_lld_unittest(lldMachOTests
  MachONormalizedFileYAMLTests.cpp
  )

target_link_libraries(lldMachOTests
  lldMachO
  lldYAML
  )
