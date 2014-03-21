add_llvm_library(LLVMProfileData
  InstrProf.cpp
  InstrProfReader.cpp

  LINK_LIBS
  LLVMSupport
)
