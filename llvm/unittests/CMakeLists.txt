add_custom_target(UnitTests)
set_target_properties(UnitTests PROPERTIES FOLDER "Tests")

function(add_llvm_unittest test_dirname)
  add_unittest(UnitTests ${test_dirname} ${ARGN})
endfunction()

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

add_llvm_unittest(ADTTests
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

add_llvm_unittest(AnalysisTests
  Analysis/ScalarEvolutionTest.cpp
  )

add_llvm_unittest(ExecutionEngineTests
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

add_llvm_unittest(ExecutionEngine/JITTests
  ${JITTestsSources}
  )

if(MINGW OR CYGWIN)
  set_property(TARGET JITTests PROPERTY LINK_FLAGS -Wl,--export-all-symbols)
endif()

add_llvm_unittest(Transforms/UtilsTests
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

add_llvm_unittest(VMCoreTests
  ${VMCoreSources}
  )

add_llvm_unittest(BitcodeTests
  Bitcode/BitReaderTest.cpp
  )

set(LLVM_LINK_COMPONENTS
  Support
  Core
  )

add_llvm_unittest(SupportTests
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
