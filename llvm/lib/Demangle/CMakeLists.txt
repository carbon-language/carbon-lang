add_llvm_library(LLVMDemangle
  ItaniumDemangle.cpp

  LINK_LIBS
  LLVMSupport
)
