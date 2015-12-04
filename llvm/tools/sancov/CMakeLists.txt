set(LLVM_LINK_COMPONENTS
  AllTargetsAsmPrinters
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  DebugInfoDWARF
  DebugInfoPDB
  Object
  Support
  Symbolize
  )

add_llvm_tool(sancov
  sancov.cc
  )
