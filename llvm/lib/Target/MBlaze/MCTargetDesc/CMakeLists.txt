add_llvm_library(LLVMMBlazeDesc
  MBlazeAsmBackend.cpp
  MBlazeMCAsmInfo.cpp
  MBlazeMCCodeEmitter.cpp
  MBlazeMCTargetDesc.cpp
  )
add_dependencies(LLVMMBlazeDesc MBlazeCommonTableGen)

target_link_libraries(LLVMMBlazeDesc LLVMMBlazeAsmPrinter)
