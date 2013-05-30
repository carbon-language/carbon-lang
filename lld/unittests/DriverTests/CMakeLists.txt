add_lld_unittest(DriverTests
  UniversalDriverTest.cpp
  WinLinkDriverTest.cpp
  )

target_link_libraries(DriverTests
  lldDriver
  )
