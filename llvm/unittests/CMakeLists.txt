function(add_llvm_unittest test_dirname)
  string(REGEX MATCH "([^/]+)$" test_name ${test_dirname})
  if (CMAKE_BUILD_TYPE)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
      ${LLVM_BINARY_DIR}/unittests/${test_dirname}/${CMAKE_BUILD_TYPE})
  else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
      ${LLVM_BINARY_DIR}/unittests/${test_dirname})
  endif()
  if( NOT LLVM_BUILD_TESTS )
    set(EXCLUDE_FROM_ALL ON)
  endif()
  add_llvm_executable(${test_name}Tests ${ARGN})
  add_dependencies(UnitTests ${test_name}Tests)
  set_target_properties(${test_name}Tests PROPERTIES FOLDER "Tests")
endfunction()

add_custom_target(UnitTests)
set_target_properties(UnitTests PROPERTIES FOLDER "Tests")

include_directories(${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include)
add_definitions(-DGTEST_HAS_RTTI=0)
if( LLVM_COMPILER_IS_GCC_COMPATIBLE )
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS "-frtti" "-fno-rtti")
elseif( MSVC )
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS "/GR" "/GR-")
endif()

if (NOT LLVM_ENABLE_THREADS)
  add_definitions(-DGTEST_HAS_PTHREAD=0)
endif()

if(SUPPORTS_NO_VARIADIC_MACROS_FLAG)
  add_definitions("-Wno-variadic-macros")
endif()

set(LLVM_LINK_COMPONENTS
  jit
  interpreter
  nativecodegen
  BitWriter
  BitReader
  AsmParser
  Core
  Support
  )

set(LLVM_USED_LIBS
  gtest
  gtest_main
  LLVMSupport # gtest needs it for raw_ostream.
  )

add_llvm_unittest(ADT
  ADT/APFloatTest.cpp
  ADT/APIntTest.cpp
  ADT/BitVectorTest.cpp
  ADT/DAGDeltaAlgorithmTest.cpp
  ADT/DeltaAlgorithmTest.cpp
  ADT/DenseMapTest.cpp
  ADT/DenseSetTest.cpp
  ADT/FoldingSet.cpp
  ADT/ilistTest.cpp
  ADT/ImmutableSetTest.cpp
  ADT/IntEqClassesTest.cpp
  ADT/IntervalMapTest.cpp
  ADT/PackedVectorTest.cpp
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

set(JITTestsSources
  ExecutionEngine/JIT/JITEventListenerTest.cpp
  ExecutionEngine/JIT/JITMemoryManagerTest.cpp
  ExecutionEngine/JIT/JITTest.cpp
  ExecutionEngine/JIT/MultiJITTest.cpp
  )

if(MSVC)
  list(APPEND JITTestsSources ExecutionEngine/JIT/JITTests.def)
endif()

add_llvm_unittest(ExecutionEngine/JIT ${JITTestsSources})

if(MINGW)
  set_property(TARGET JITTests PROPERTY LINK_FLAGS -Wl,--export-all-symbols)
endif()

add_llvm_unittest(Transforms/Utils
  Transforms/Utils/Cloning.cpp
  )

set(VMCoreSources
  VMCore/ConstantsTest.cpp
  VMCore/InstructionsTest.cpp
  VMCore/MetadataTest.cpp
  VMCore/PassManagerTest.cpp
  VMCore/ValueMapTest.cpp
  VMCore/VerifierTest.cpp
  )

# MSVC9 and 8 cannot compile ValueMapTest.cpp due to their bug.
# See issue#331418 in Visual Studio.
if(MSVC AND MSVC_VERSION LESS 1600)
  list(REMOVE_ITEM VMCoreSources VMCore/ValueMapTest.cpp)
endif()

add_llvm_unittest(VMCore ${VMCoreSources})

set(LLVM_LINK_COMPONENTS
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
  Support/Path.cpp
  Support/raw_ostream_test.cpp
  Support/RegexTest.cpp
  Support/SwapByteOrderTest.cpp
  Support/TimeValue.cpp
  Support/TypeBuilderTest.cpp
  Support/ValueHandleTest.cpp
  )
