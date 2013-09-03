set(LLVM_LINK_COMPONENTS
  asmprinter
  codegen
  support
  )

set(DebugInfoSources
  DIEHashTest.cpp
  )

add_llvm_unittest(DebugInfoTests
  ${DebugInfoSources}
  )
