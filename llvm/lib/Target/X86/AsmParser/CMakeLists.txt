add_llvm_library(LLVMX86AsmParser
  X86AsmInstrumentation.cpp
  X86AsmParser.cpp

  LINK_LIBS
  LLVMX86CodeGen
  )
