if(HAVE_LIBEDIT)
  set(link_libs edit)
endif()

add_llvm_library(LLVMLineEditor
  LineEditor.cpp

  LINK_LIBS
  LLVMSupport
  ${link_libs}
)
