add_lld_library(lldARMELFTarget
  ARMLinkingContext.cpp
  ARMTargetHandler.cpp
  ARMRelocationHandler.cpp
  ARMRelocationPass.cpp
  )

target_link_libraries(lldARMELFTarget ${cmake_2_8_12_INTERFACE}
  lldCore
  )
