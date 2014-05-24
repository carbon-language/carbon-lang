set(LLVM_NO_RTTI 1)

if (PLUTO_FOUND)
  set(POLLY_PLUTO_FILES Pluto.cpp)
endif (PLUTO_FOUND)

set(POLLY_JSON_FILES
    JSON/json_reader.cpp
    JSON/json_value.cpp
    JSON/json_writer.cpp
)

if (CLOOG_FOUND)
  set(CLOOG_FILES
      CodeGen/Cloog.cpp
      CodeGen/CodeGeneration.cpp)
endif (CLOOG_FOUND)

set(ISL_CODEGEN_FILES
    CodeGen/IslAst.cpp
    CodeGen/IslCodeGeneration.cpp)

if (GPU_CODEGEN)
  set (GPGPU_CODEGEN_FILES
       CodeGen/PTXGenerator.cpp)
endif (GPU_CODEGEN)

if (SCOPLIB_FOUND)
  set(POLLY_SCOPLIB_FILES
      Transform/Pocc.cpp
      Exchange/ScopLib.cpp
      Exchange/ScopLibExporter.cpp
      Exchange/ScopLibImporter.cpp)
endif (SCOPLIB_FOUND)

add_polly_library(Polly
  Analysis/Dependences.cpp
  Analysis/ScopDetection.cpp
  Analysis/ScopDetectionDiagnostic.cpp
  Analysis/ScopInfo.cpp
  Analysis/ScopGraphPrinter.cpp
  Analysis/ScopPass.cpp
  Analysis/TempScopInfo.cpp
  CodeGen/BlockGenerators.cpp
  ${CLOOG_FILES}
  ${ISL_CODEGEN_FILES}
  CodeGen/LoopGenerators.cpp
  CodeGen/IRBuilder.cpp
  CodeGen/Utils.cpp
  ${GPGPU_CODEGEN_FILES}
  Exchange/JSONExporter.cpp
  ${POLLY_SCOPLIB_FILES}
  Support/GICHelper.cpp
  Support/SCEVValidator.cpp
  Support/RegisterPasses.cpp
  Support/ScopHelper.cpp
  ${POLLY_JSON_FILES}
  Transform/Canonicalization.cpp
  Transform/CodePreparation.cpp
  Transform/DeadCodeElimination.cpp
  Transform/IndependentBlocks.cpp
  Transform/IndVarSimplify.cpp
  Transform/ScheduleOptimizer.cpp
  ${POLLY_PLUTO_FILES}
  )

# Build a monolithic Polly.a and a thin module LLVMPolly.moduleext that links to
# that static library.
add_polly_loadable_module(LLVMPolly
  Polly.cpp
)

if (TARGET intrinsics_gen)
  # Check if we are building as part of an LLVM build
  add_dependencies(Polly intrinsics_gen)
endif()

target_link_libraries(LLVMPolly Polly)

set_target_properties(LLVMPolly
  PROPERTIES
  LINKER_LANGUAGE CXX
  PREFIX "")
