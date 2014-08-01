set(LLVM_LINK_COMPONENTS
  IRReader
  BitWriter
  Core
  Support
  )

add_llvm_unittest(BitcodeTests
  BitReaderTest.cpp
  )
