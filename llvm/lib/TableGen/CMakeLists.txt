## FIXME: This only requires RTTI because tblgen uses it.  Fix that.
set(LLVM_REQUIRES_RTTI 1)
set(LLVM_REQUIRES_EH 1)

add_llvm_library(LLVMTableGen
  Error.cpp
  Main.cpp
  Record.cpp
  TableGenBackend.cpp
  TGLexer.cpp
  TGParser.cpp
  )

add_llvm_library_dependencies(LLVMTableGen
  LLVMSupport
  )
