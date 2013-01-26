add_lld_library(lldX86_64ELFTarget
  X86_64TargetInfo.cpp
  )

target_link_libraries(lldX86_64ELFTarget
  lldCore
  )
