
if (LLVM_ENABLE_SPHINX)
  if (SPHINX_FOUND)
    include(AddSphinxTarget)
    if (${SPHINX_OUTPUT_HTML})
      add_sphinx_target(html libcxx)
    endif()
  endif()
endif()