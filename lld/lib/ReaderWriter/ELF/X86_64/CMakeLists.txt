add_lld_library(lldX86_64ELFTarget
  X86_64LinkingContext.cpp
  X86_64TargetHandler.cpp
  X86_64RelocationHandler.cpp
  X86_64RelocationPass.cpp
  )

target_link_libraries(lldX86_64ELFTarget ${cmake_2_8_12_INTERFACE}
  lldCore
  )
