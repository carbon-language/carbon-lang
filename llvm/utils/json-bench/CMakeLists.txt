add_llvm_utility(json-bench
  JSONBench.cpp
  )

target_link_libraries(json-bench LLVMSupport)
