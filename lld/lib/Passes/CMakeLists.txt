add_lld_library(lldPasses
  LayoutPass.cpp
  RoundTripNativePass.cpp
  RoundTripYAMLPass.cpp
  )

target_link_libraries(lldPasses ${cmake_2_8_12_INTERFACE}
  lldReaderWriter
  )
