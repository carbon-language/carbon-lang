include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/X86
  ${LLVM_BINARY_DIR}/lib/Target/X86
  )

add_llvm_library(LLVMExegesisX86
  Target.cpp

  STATIC

  LINK_COMPONENTS
  X86
  Exegesis
  )
