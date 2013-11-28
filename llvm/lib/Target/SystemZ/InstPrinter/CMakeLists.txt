add_llvm_library(LLVMSystemZAsmPrinter
  SystemZInstPrinter.cpp
  )

add_dependencies(LLVMSystemZAsmPrinter SystemZCommonTableGen)
