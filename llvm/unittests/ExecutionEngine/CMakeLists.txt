set(LLVM_LINK_COMPONENTS
  jit
  interpreter
  nativecodegen
  )

add_llvm_unittest(ExecutionEngineTests
  ExecutionEngineTest.cpp
  )

add_subdirectory(JIT)
