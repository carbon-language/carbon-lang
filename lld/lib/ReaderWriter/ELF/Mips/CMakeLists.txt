add_lld_library(lldMipsELFTarget
  MipsCtorsOrderPass.cpp
  MipsLinkingContext.cpp
  MipsRelocationHandler.cpp
  MipsRelocationPass.cpp
  MipsTargetHandler.cpp
  )

target_link_libraries(lldMipsELFTarget
  lldCore
  )
