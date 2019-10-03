set(LLVM_LINK_COMPONENTS
  TableGen
  Support
  )

add_llvm_unittest(TableGenTests
  CodeExpanderTest.cpp
  $<TARGET_OBJECTS:obj.LLVMTableGenGlobalISel>
  )

include_directories(${CMAKE_SOURCE_DIR}/utils/TableGen)
