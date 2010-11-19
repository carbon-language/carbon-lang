function(add_llvm_unittest test_name)
  if (CMAKE_BUILD_TYPE)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
      ${LLVM_BINARY_DIR}/unittests/${test_name}/${CMAKE_BUILD_TYPE})
  else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
      ${LLVM_BINARY_DIR}/unittests/${test_name})
  endif()
  if( NOT LLVM_BUILD_TESTS )
    set(EXCLUDE_FROM_ALL ON)
  endif()
  add_llvm_executable(${test_name}Tests ${ARGN})
  add_dependencies(UnitTests ${test_name}Tests)
endfunction()

add_custom_target(UnitTests)

include_directories(${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include)
add_definitions(-DGTEST_HAS_RTTI=0)

set(LLVM_LINK_COMPONENTS
  jit
  interpreter
  nativecodegen
  BitWriter
  BitReader
  AsmParser
  Core
  System
  Support
  )

set(LLVM_USED_LIBS
  gtest
  gtest_main
  )

add_llvm_unittest(ADT
  ADT/APFloatTest.cpp
  ADT/APIntTest.cpp
  ADT/BitVectorTest.cpp
  ADT/DAGDeltaAlgorithmTest.cpp
  ADT/DeltaAlgorithmTest.cpp
  ADT/DenseMapTest.cpp
  ADT/DenseSetTest.cpp
  ADT/ilistTest.cpp
  ADT/ImmutableSetTest.cpp
  ADT/IntervalMapTest.cpp
  ADT/SmallBitVectorTest.cpp
  ADT/SmallStringTest.cpp
  ADT/SmallVectorTest.cpp
  ADT/SparseBitVectorTest.cpp
  ADT/StringMapTest.cpp
  ADT/StringRefTest.cpp
  ADT/TripleTest.cpp
  ADT/TwineTest.cpp
 )

add_llvm_unittest(Analysis
  Analysis/ScalarEvolutionTest.cpp
  )

add_llvm_unittest(ExecutionEngine
  ExecutionEngine/ExecutionEngineTest.cpp
  )

add_llvm_unittest(JIT
  ExecutionEngine/JIT/JITEventListenerTest.cpp
  ExecutionEngine/JIT/JITMemoryManagerTest.cpp
  ExecutionEngine/JIT/JITTest.cpp
  ExecutionEngine/JIT/MultiJITTest.cpp
  )

add_llvm_unittest(Transforms
  Transforms/Utils/Cloning.cpp
  )

add_llvm_unittest(VMCore
  VMCore/ConstantsTest.cpp
  VMCore/DerivedTypesTest.cpp
  VMCore/InstructionsTest.cpp
  VMCore/MetadataTest.cpp
  VMCore/PassManagerTest.cpp
  VMCore/ValueMapTest.cpp
  VMCore/VerifierTest.cpp
  )

set(LLVM_LINK_COMPONENTS
  System
  Support
  Core
  )

add_llvm_unittest(Support
  Support/AllocatorTest.cpp
  Support/Casting.cpp
  Support/CommandLineTest.cpp
  Support/ConstantRangeTest.cpp
  Support/EndianTest.cpp
  Support/LeakDetectorTest.cpp
  Support/MathExtrasTest.cpp
  Support/raw_ostream_test.cpp
  Support/RegexTest.cpp
  Support/SwapByteOrderTest.cpp
  Support/System.cpp
  Support/TypeBuilderTest.cpp
  Support/ValueHandleTest.cpp
  )
