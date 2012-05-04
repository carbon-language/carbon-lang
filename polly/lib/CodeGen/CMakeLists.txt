if (CLOOG_FOUND)
  set(CLOOG_FILES
      Cloog.cpp
      CodeGeneration.cpp)
endif (CLOOG_FOUND)
add_polly_library(PollyCodeGen
  BlockGenerators.cpp
  ${CLOOG_FILES}
  LoopGenerators.cpp
)
