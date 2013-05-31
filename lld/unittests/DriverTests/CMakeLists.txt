add_lld_unittest(DriverTests
  UniversalDriverTest.cpp
  GnuLdDriverTest.cpp
  WinLinkDriverTest.cpp
  )

target_link_libraries(DriverTests
  lldDriver
  )
