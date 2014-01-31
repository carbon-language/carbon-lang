add_llvm_library(LLVMLineEditor
  LineEditor.cpp
  )

if(HAVE_LIBEDIT)
  target_link_libraries(LLVMLineEditor edit)
endif()
