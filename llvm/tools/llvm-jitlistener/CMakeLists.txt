# This tool is excluded from the CMake build if Intel JIT events are disabled.

link_directories( ${LLVM_INTEL_JITEVENTS_LIBDIR} )
include_directories( ${LLVM_INTEL_JITEVENTS_INCDIR} )

set(LLVM_LINK_COMPONENTS
  asmparser
  bitreader
  inteljitevents
  interpreter
  jit
  mcjit
  nativecodegen
  object
  selectiondag
  )

add_llvm_tool(llvm-jitlistener
  llvm-jitlistener.cpp
  )
