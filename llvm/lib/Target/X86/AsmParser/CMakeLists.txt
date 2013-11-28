add_llvm_library(LLVMX86AsmParser
  X86AsmParser.cpp
  )

add_dependencies(LLVMX86AsmParser X86CommonTableGen)
