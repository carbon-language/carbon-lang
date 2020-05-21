add_llvm_library(LLVMBOLTRuntimeLibs
  RuntimeLibrary.cpp
  InstrumentationRuntimeLibrary.cpp

  DEPENDS
  intrinsics_gen
  )

include_directories( ${LLVM_MAIN_SRC_DIR}/tools/llvm-bolt/src )
