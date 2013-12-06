set(LLVM_LINK_COMPONENTS
  TransformUtils
  )

add_llvm_unittest(UtilsTests
  ASanStackFrameLayoutTest.cpp
  Cloning.cpp
  IntegerDivision.cpp
  Local.cpp
  SpecialCaseList.cpp
  )
