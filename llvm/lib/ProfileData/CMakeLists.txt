add_llvm_library(LLVMProfileData
  InstrProf.cpp
  InstrProfReader.cpp
  InstrProfWriter.cpp

  LINK_LIBS
  LLVMSupport
)
