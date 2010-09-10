# AsmParser
add_llvm_library(LLVMAsmParser
  LLLexer.cpp
  LLParser.cpp
  Parser.cpp
  )

target_link_libraries(LLVMAsmParser LLVMCore)
