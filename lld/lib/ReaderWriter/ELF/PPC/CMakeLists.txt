add_lld_library(lldPPCELFTarget
  PPCTargetHandler.cpp
  PPCTargetInfo.cpp
  )

target_link_libraries(lldPPCELFTarget
  lldCore
  )
