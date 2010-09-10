add_llvm_example(ParallelJIT
  ParallelJIT.cpp
  )

add_llvm_link_components(ParallelJIT jit interpreter nativecodegen)

if(HAVE_LIBPTHREAD)
  target_link_libraries(ParallelJIT pthread)
endif(HAVE_LIBPTHREAD)
