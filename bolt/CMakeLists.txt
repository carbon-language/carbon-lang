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

add_llvm_tool(llvm-flo
  llvm-flo.cpp
  )
