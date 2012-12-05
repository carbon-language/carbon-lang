add_llvm_library(LLVMOption
  Arg.cpp
  ArgList.cpp
  Option.cpp
  OptTable.cpp
  )

target_link_libraries(LLVMOption LLVMSupport)
