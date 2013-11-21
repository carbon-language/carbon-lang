add_lld_unittest(DriverTests
  UniversalDriverTest.cpp
  GnuLdDriverTest.cpp
  DarwinLdDriverTest.cpp
  WinLinkDriverTest.cpp
  InputGraphTest.cpp
  )

target_link_libraries(DriverTests
  lldDriver
  )
