add_subdirectory(merge-fdata)

set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  CodeGen
  Core
  DebugInfoDWARF
  MC
  MCDisassembler
  MCParser
  Object
  Orcjit
  Support
  )

add_llvm_tool(llvm-bolt
  llvm-bolt.cpp
  BinaryBasicBlock.cpp
  BinaryContext.cpp
  BinaryFunction.cpp
  BinaryPasses.cpp
  BinaryPassManager.cpp
  DataReader.cpp
  DebugData.cpp
  Exceptions.cpp
  RewriteInstance.cpp
  )
