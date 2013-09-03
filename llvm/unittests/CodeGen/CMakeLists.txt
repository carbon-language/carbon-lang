set(LLVM_LINK_COMPONENTS
  asmprinter
  codegen
  support
  )

set(CodeGenSources
  DIEHashTest.cpp
  )

add_llvm_unittest(CodeGenTests
  ${CodeGenSources}
  )
