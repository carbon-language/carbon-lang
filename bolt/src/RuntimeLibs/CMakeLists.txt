add_llvm_library(LLVMBOLTRuntimeLibs
  RuntimeLibrary.cpp
  HugifyRuntimeLibrary.cpp
  InstrumentationRuntimeLibrary.cpp

  DEPENDS
  intrinsics_gen
  )

include_directories( ${BOLT_SOURCE_DIR}/src )
