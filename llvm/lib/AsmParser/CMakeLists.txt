# AsmParser
add_llvm_library(LLVMAsmParser
  LLLexer.cpp
  LLParser.cpp
  Parser.cpp
  )

add_llvm_library_dependencies(LLVMAsmParser
  LLVMCore
  LLVMSupport
  LLVMTarget
  )
