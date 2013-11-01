add_lld_library(lldPasses
  GOTPass.cpp
  StubsPass.cpp
  LayoutPass.cpp
  RoundTripNativePass.cpp
  RoundTripYAMLPass.cpp
  )

target_link_libraries(lldPasses lldReaderWriter)
