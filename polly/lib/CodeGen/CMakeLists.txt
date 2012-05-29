if (CLOOG_FOUND)
  set(CLOOG_FILES
      Cloog.cpp
      CodeGeneration.cpp)
endif (CLOOG_FOUND)

if (ISL_CODEGEN_FOUND)
  set(ISL_CODEGEN_FILES
      IslCodeGeneration.cpp)
endif (ISL_CODEGEN_FOUND)

add_polly_library(PollyCodeGen
  BlockGenerators.cpp
  ${CLOOG_FILES}
  ${ISL_CODEGEN_FILES}
  LoopGenerators.cpp
  Utils.cpp
)
