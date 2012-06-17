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

# Visual Studio 2012 only supports up to 8 template parameters in
# std::tr1::tuple by default, but gtest requires 10
if(MSVC AND MSVC_VERSION EQUAL 1700)
  add_definitions(-D_VARIADIC_MAX=10)
endif ()

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
  ADT/HashingTest.cpp
  ADT/ilistTest.cpp
  ADT/ImmutableSetTest.cpp
  ADT/IntEqClassesTest.cpp
  ADT/IntervalMapTest.cpp
  ADT/IntrusiveRefCntPtrTest.cpp
  ADT/PackedVectorTest.cpp
  ADT/SCCIteratorTest.cpp
  ADT/SmallPtrSetTest.cpp
  ADT/SmallStringTest.cpp
  ADT/SmallVectorTest.cpp
  ADT/SparseBitVectorTest.cpp
  ADT/SparseSetTest.cpp
  ADT/StringMapTest.cpp
  ADT/StringRefTest.cpp
  ADT/TripleTest.cpp
  ADT/TwineTest.cpp
  ADT/VariadicFunctionTest.cpp
 )

add_llvm_unittest(Analysis
  Analysis/ScalarEvolutionTest.cpp
  )

add_llvm_unittest(ExecutionEngine
  ExecutionEngine/ExecutionEngineTest.cpp
  )

if( LLVM_USE_INTEL_JITEVENTS )
  include_directories( ${LLVM_INTEL_JITEVENTS_INCDIR} )
  link_directories( ${LLVM_INTEL_JITEVENTS_LIBDIR} )
  set(ProfileTestSources
    ExecutionEngine/JIT/IntelJITEventListenerTest.cpp
    )
  set(LLVM_LINK_COMPONENTS
    ${LLVM_LINK_COMPONENTS}
    IntelJITEvents
    ) 
endif( LLVM_USE_INTEL_JITEVENTS )

if( LLVM_USE_OPROFILE )
  set(ProfileTestSources
    ${ProfileTestSources}
    ExecutionEngine/JIT/OProfileJITEventListenerTest.cpp
    )
  set(LLVM_LINK_COMPONENTS
    ${LLVM_LINK_COMPONENTS}
    OProfileJIT
    )
endif( LLVM_USE_OPROFILE )

set(JITTestsSources
  ExecutionEngine/JIT/JITEventListenerTest.cpp
  ExecutionEngine/JIT/JITMemoryManagerTest.cpp
  ExecutionEngine/JIT/JITTest.cpp
  ExecutionEngine/JIT/MultiJITTest.cpp
  ${ProfileTestSources}
  )

if(MSVC)
  list(APPEND JITTestsSources ExecutionEngine/JIT/JITTests.def)
endif()

add_llvm_unittest(ExecutionEngine/JIT ${JITTestsSources})

if(MINGW OR CYGWIN)
  set_property(TARGET JITTests PROPERTY LINK_FLAGS -Wl,--export-all-symbols)
endif()

add_llvm_unittest(Transforms/Utils
  Transforms/Utils/Cloning.cpp
  )

set(VMCoreSources
  VMCore/ConstantsTest.cpp
  VMCore/DominatorTreeTest.cpp
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

add_llvm_unittest(Bitcode
  Bitcode/BitReaderTest.cpp
  )

set(LLVM_LINK_COMPONENTS
  Support
  Core
  )

add_llvm_unittest(Support
  Support/AlignOfTest.cpp
  Support/AllocatorTest.cpp
  Support/BlockFrequencyTest.cpp
  Support/Casting.cpp
  Support/CommandLineTest.cpp
  Support/ConstantRangeTest.cpp
  Support/DataExtractorTest.cpp
  Support/EndianTest.cpp
  Support/IntegersSubsetTest.cpp
  Support/IRBuilderTest.cpp
  Support/LeakDetectorTest.cpp
  Support/ManagedStatic.cpp
  Support/MathExtrasTest.cpp
  Support/MDBuilderTest.cpp
  Support/Path.cpp
  Support/raw_ostream_test.cpp
  Support/RegexTest.cpp
  Support/SwapByteOrderTest.cpp
  Support/TimeValue.cpp
  Support/TypeBuilderTest.cpp
  Support/ValueHandleTest.cpp
  Support/YAMLParserTest.cpp
  )
