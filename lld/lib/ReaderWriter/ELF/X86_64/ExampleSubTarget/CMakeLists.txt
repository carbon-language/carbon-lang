add_llvm_library(lldExampleSubTarget
  ExampleLinkingContext.cpp
  ExampleTargetHandler.cpp
  LINK_LIBS
    lldX86_64ELFTarget
  )
