set(LLVM_LINK_COMPONENTS
  HexagonCodeGen
  HexagonDesc
  HexagonInfo
  MC
  Support
  )

include_directories (${LLVM_MAIN_SRC_DIR}/lib/Target/Hexagon)
include_directories (${LLVM_BINARY_DIR}/lib/Target/Hexagon)

add_llvm_unittest(HexagonTests
  HexagonMCCodeEmitterTest.cpp
  )
