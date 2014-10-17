add_lld_library(lldPPCELFTarget
  PPCLinkingContext.cpp
  PPCTargetHandler.cpp
  )

target_link_libraries(lldPPCELFTarget ${cmake_2_8_12_INTERFACE}
  lldCore
  )
