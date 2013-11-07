set(LLVM_LINK_COMPONENTS
  core
  linker
  )

set(LinkerSources
  LinkModulesTest.cpp
  )

add_llvm_unittest(LinkerTests
  ${LinkerSources}
  )
