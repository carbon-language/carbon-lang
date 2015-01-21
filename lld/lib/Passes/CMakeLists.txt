add_llvm_library(lldPasses
  LayoutPass.cpp
  RoundTripNativePass.cpp
  RoundTripYAMLPass.cpp
  LINK_LIBS
    lldReaderWriter
    LLVMSupport
  )
