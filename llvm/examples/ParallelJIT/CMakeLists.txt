set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  Interpreter
  MC
  Support
  nativecodegen
  )

add_llvm_example(ParallelJIT
  ParallelJIT.cpp
  )

if(HAVE_LIBPTHREAD)
  target_link_libraries(ParallelJIT pthread)
endif(HAVE_LIBPTHREAD)
