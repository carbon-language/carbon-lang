set(LLVM_LINK_COMPONENTS
  interpreter
  )

add_llvm_unittest(ExecutionEngineTests
  ExecutionEngineTest.cpp
  )

# Include JIT/MCJIT tests only if native arch is a JIT target.
list(FIND LLVM_TARGETS_WITH_JIT "${LLVM_NATIVE_ARCH}" have_jit)
if (NOT have_jit EQUAL -1 )
  add_subdirectory(JIT)
  add_subdirectory(MCJIT)
endif()
