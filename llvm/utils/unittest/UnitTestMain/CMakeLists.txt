add_llvm_library(gtest_main
  TestMain.cpp
  )

# Depends on llvm::cl
target_link_libraries(gtest_main
  gtest
  LLVMSupport
  )
