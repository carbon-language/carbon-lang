add_llvm_library(LLVMMC
  DIContext.cpp
  DWARFAbbreviationDeclaration.cpp
  DWARFCompileUnit.cpp
  DWARFContext.cpp
  DWARFDebugAbbrev.cpp
  DWARFDebugInfoEntry.cpp
  DWARFFormValue.cpp
  )

add_llvm_library_dependencies(LLVMDebugInfo
  LLVMSupport
  )
