
include_directories( ${LLVM_INTEL_JITEVENTS_INCDIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. )

set(system_libs
  ${system_libs}
  jitprofiling
  )

add_llvm_library(LLVMIntelJITEvents
  IntelJITEventListener.cpp
  )
