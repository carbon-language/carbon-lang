# build a monolithic libLLVMPollyLib.$shlibext
# and a thin module LLVMPolly.moduleext that links to that shared library

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

if (OPENSCOP_FOUND)
  set(POLLY_OPENSCOP_FILES
      Exchange/OpenScopImporter.cpp
      Exchange/OpenScopExporter.cpp)
endif (OPENSCOP_FOUND)

if (SCOPLIB_FOUND)
  set(POLLY_SCOPLIB_FILES
      Pocc.cpp
      Exchange/ScopLib.cpp
      Exchange/ScopLibExporter.cpp
      Exchange/ScopLibImporter.cpp)
endif (SCOPLIB_FOUND)

add_polly_library(LLVMPollyLib
  Analysis/Dependences.cpp
  Analysis/ScopDetection.cpp
  Analysis/ScopInfo.cpp
  Analysis/ScopGraphPrinter.cpp
  Analysis/ScopPass.cpp
  Analysis/TempScopInfo.cpp
  CodeGen/BlockGenerators.cpp
  ${CLOOG_FILES}
  ${ISL_CODEGEN_FILES}
  CodeGen/LoopGenerators.cpp
  CodeGen/Utils.cpp
  ${GPGPU_CODEGEN_FILES}
  Support/GICHelper.cpp
  Support/SCEVValidator.cpp
  Support/ScopHelper.cpp
  Exchange/JSONExporter.cpp
  ${POLLY_JSON_FILES}
  ${POLLY_OPENSCOP_FILES}
  Canonicalization.cpp
  CodePreparation.cpp
  DeadCodeElimination.cpp
  IndependentBlocks.cpp
  IndVarSimplify.cpp
  Pocc.cpp
  RegisterPasses.cpp
  ScheduleOptimizer.cpp
  ${POLLY_SCOPLIB_FILES}
  ${POLLY_PLUTO_FILES}
  )

add_polly_loadable_module(LLVMPolly
  PollyModule.cpp
)

if (TARGET intrinsics_gen)
  # Check if we are building as part of an LLVM build
  add_dependencies(LLVMPolly intrinsics_gen)
endif()

target_link_libraries(LLVMPolly LLVMPollyLib)

set_target_properties(LLVMPolly
  PROPERTIES
  LINKER_LANGUAGE CXX
  PREFIX "")

if (PLUTO_FOUND)
  target_link_libraries(LLVMPolly ${PLUTO_LIBRARY})
endif(PLUTO_FOUND)
