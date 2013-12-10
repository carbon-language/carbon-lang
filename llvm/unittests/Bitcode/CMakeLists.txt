set(LLVM_LINK_COMPONENTS
  BitReader
  BitWriter
  Core
  Support
  )

add_llvm_unittest(BitcodeTests
  BitReaderTest.cpp
  )
