add_lld_library(lldMipsELFTarget
  MipsLinkingContext.cpp
  MipsRelocationHandler.cpp
  MipsTargetHandler.cpp
  )

target_link_libraries(lldMipsELFTarget
  lldCore
  )
