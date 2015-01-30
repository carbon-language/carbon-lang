set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  DebugInfoDWARF
  MC
  MCDisassembler
  Object
  Support
  )

add_llvm_tool(llvm-objdump
  llvm-objdump.cpp
  COFFDump.cpp
  ELFDump.cpp
  MachODump.cpp
  )
