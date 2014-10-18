add_lld_library(lldX86ELFTarget
  X86LinkingContext.cpp
  X86TargetHandler.cpp
  X86RelocationHandler.cpp
  )

target_link_libraries(lldX86ELFTarget ${cmake_2_8_12_INTERFACE}
  lldCore
  )
