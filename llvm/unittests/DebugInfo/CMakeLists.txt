set(LLVM_LINK_COMPONENTS
  DebugInfo
  )

set(DebugInfoSources
  DWARFFormValueTest.cpp
  )

add_llvm_unittest(DebugInfoTests
  ${DebugInfoSources}
  )
