add_llvm_library(gtest_main
  TestMain.cpp

  LINK_LIBS
  gtest
  LLVMSupport # Depends on llvm::cl
  )
