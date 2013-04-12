set(LLVM_LINK_COMPONENTS
  asmparser
  core
  ipa
  )

set(IRSources
  AttributesTest.cpp
  ConstantsTest.cpp
  DominatorTreeTest.cpp
  IRBuilderTest.cpp
  InstructionsTest.cpp
  MDBuilderTest.cpp
  MetadataTest.cpp
  PassManagerTest.cpp
  TypeBuilderTest.cpp
  TypesTest.cpp
  ValueMapTest.cpp
  ValueTest.cpp
  VerifierTest.cpp
  WaymarkTest.cpp
  )

# MSVC9 and 8 cannot compile ValueMapTest.cpp due to their bug.
# See issue#331418 in Visual Studio.
if(MSVC AND MSVC_VERSION LESS 1600)
  list(REMOVE_ITEM IRSources ValueMapTest.cpp)
endif()

# HACK: Declare a couple of source files as optionally compiled to satisfy the
# missing-file-checker in LLVM's weird CMake build.
set(LLVM_OPTIONAL_SOURCES
  ValueMapTest.cpp
  )

add_llvm_unittest(IRTests
  ${IRSources}
  )
