if( LLVM_USE_SANITIZE_COVERAGE )
  set(LLVM_LINK_COMPONENTS
      AsmParser
      BitWriter
      Core
      Support
      )
  add_llvm_tool(llvm-as-fuzzer 
                llvm-as-fuzzer.cpp)
  target_link_libraries(llvm-as-fuzzer
                        LLVMFuzzer
                        )
endif()
