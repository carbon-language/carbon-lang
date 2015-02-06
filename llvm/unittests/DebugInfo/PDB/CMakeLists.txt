set(LLVM_LINK_COMPONENTS
  DebugInfoPDB
  )

set(DebugInfoPDBSources
  PDBApiTest.cpp
  )

add_llvm_unittest(DebugInfoPDBTests
  ${DebugInfoPDBSources}
  )
