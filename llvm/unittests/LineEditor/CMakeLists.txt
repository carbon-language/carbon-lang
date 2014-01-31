set(LLVM_LINK_COMPONENTS
  LineEditor
  )

add_llvm_unittest(LineEditorTests
  LineEditor.cpp
  )
