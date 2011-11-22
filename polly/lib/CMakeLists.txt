add_subdirectory(Analysis)
add_subdirectory(Exchange)
add_subdirectory(Support)
add_subdirectory(JSON)

set(MODULE TRUE)
set(LLVM_NO_RTTI 1)

if (SCOPLIB_FOUND)
  set(POLLY_SCOPLIB_FILES Pocc.cpp)
endif (SCOPLIB_FOUND)

set(LLVM_USED_LIBS
  PollyAnalysis
  PollyExchange
  PollySupport
  PollyJSON
  )

add_polly_library(LLVMPolly
  Cloog.cpp
  CodePreparation.cpp
  CodeGeneration.cpp
  IndependentBlocks.cpp
  MayAliasSet.cpp
  Pocc.cpp
  RegionSimplify.cpp
  RegisterPasses.cpp
  ScheduleOptimizer.cpp
  ${POLLY_SCOPLIB_FILES}
  )

add_dependencies(LLVMPolly
  PollyAnalysis
  PollyExchange
  PollySupport
  PollyJSON
  )

set_target_properties(LLVMPolly
  PROPERTIES
  LINKER_LANGUAGE CXX
  PREFIX "")
