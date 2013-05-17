add_lld_library(lldX86_64ELFTarget
  X86_64TargetInfo.cpp
  X86_64TargetHandler.cpp
  X86_64RelocationHandler.cpp
  )

target_link_libraries(lldX86_64ELFTarget
  lldCore
  )
