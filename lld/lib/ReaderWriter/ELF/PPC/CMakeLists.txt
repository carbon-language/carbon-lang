add_llvm_library(lldPPCELFTarget
  PPCLinkingContext.cpp
  PPCTargetHandler.cpp
  LINK_LIBS
    lldCore
    LLVMSupport
  )
