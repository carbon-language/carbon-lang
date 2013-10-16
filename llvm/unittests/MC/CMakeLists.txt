set(LLVM_LINK_COMPONENTS
  MC
  )

set(MCSources
  MCAtomTest.cpp
  )

add_llvm_unittest(MCTests
  ${MCSources}
  )
