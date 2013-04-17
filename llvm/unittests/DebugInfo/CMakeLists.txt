set(LLVM_LINK_COMPONENTS
  debuginfo
  object
  support
  )

set(DebugInfoSources
  DWARFFormValueTest.cpp
  )

add_llvm_unittest(DebugInfoTests
  ${DebugInfoSources}
  )
