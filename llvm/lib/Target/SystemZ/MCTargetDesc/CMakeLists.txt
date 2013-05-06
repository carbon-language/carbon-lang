add_llvm_library(LLVMSystemZDesc
  SystemZMCAsmBackend.cpp
  SystemZMCAsmInfo.cpp
  SystemZMCCodeEmitter.cpp
  SystemZMCObjectWriter.cpp
  SystemZMCTargetDesc.cpp
  )

add_dependencies(LLVMSystemZDesc SystemZCommonTableGen)
