add_llvm_library(LLVMX86AsmPrinter
  X86ATTInstPrinter.cpp
  X86IntelInstPrinter.cpp
  X86InstComments.cpp
  )

add_dependencies(LLVMX86AsmPrinter X86CommonTableGen)
