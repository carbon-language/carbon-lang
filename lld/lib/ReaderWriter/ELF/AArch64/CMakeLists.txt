add_lld_library(lldAArch64ELFTarget
  AArch64LinkingContext.cpp
  AArch64TargetHandler.cpp
  AArch64RelocationHandler.cpp
  AArch64RelocationPass.cpp
  )

target_link_libraries(lldAArch64ELFTarget
  lldCore
  )
