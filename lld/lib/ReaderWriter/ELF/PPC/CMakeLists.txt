add_lld_library(lldPPCELFTarget
  PPCLinkingContext.cpp
  PPCTargetHandler.cpp
  )

target_link_libraries(lldPPCELFTarget
  lldCore
  )
