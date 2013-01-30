add_lld_library(lldX86ELFTarget
  X86TargetInfo.cpp
  X86TargetHandler.cpp
  )

target_link_libraries(lldX86ELFTarget
  lldCore
  )
