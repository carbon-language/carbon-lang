set(LLVM_LINK_COMPONENTS
  asmparser
  bitreader
  bitwriter
  jit
  mcjit
  nativecodegen
  )

set(MCJITTestsSources
  MCJITTest.cpp
  MCJITMemoryManagerTest.cpp
  MCJITObjectCacheTest.cpp
  )

if(MSVC)
  list(APPEND MCJITTestsSources MCJITTests.def)
endif()

add_llvm_unittest(MCJITTests
  ${MCJITTestsSources}
  )

if(MINGW OR CYGWIN)
  set_property(TARGET MCJITTests PROPERTY LINK_FLAGS -Wl,--export-all-symbols)
endif()
