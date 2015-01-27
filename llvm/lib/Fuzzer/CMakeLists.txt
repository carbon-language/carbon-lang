add_library(LLVMFuzzer STATIC
  EXCLUDE_FROM_ALL  # Do not build if you are not building fuzzers.
  FuzzerCrossOver.cpp
  FuzzerIO.cpp
  FuzzerLoop.cpp
  FuzzerMain.cpp
  FuzzerMutate.cpp
  FuzzerUtil.cpp
  )
