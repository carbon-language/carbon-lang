set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  CodeGen
  Core
  DebugInfoDWARF
  MC
  MIRParser
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
