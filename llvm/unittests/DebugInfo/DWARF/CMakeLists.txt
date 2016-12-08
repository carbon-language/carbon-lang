set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  AsmPrinter
  CodeGen
  Core
  DebugInfoDWARF
  MC
  MIRParser
  Object
  Support
  Target
  )

set(DebugInfoSources
  DwarfGenerator.cpp
  DWARFDebugInfoTest.cpp
  DWARFFormValueTest.cpp
  )

add_llvm_unittest(DebugInfoDWARFTests
  ${DebugInfoSources}
  )
