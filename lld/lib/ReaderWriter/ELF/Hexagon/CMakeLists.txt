add_lld_library(lldHexagonELFTarget
  HexagonLinkingContext.cpp
  HexagonRelocationHandler.cpp
  HexagonTargetHandler.cpp
  )

target_link_libraries(lldHexagonELFTarget ${cmake_2_8_12_INTERFACE}
  lldCore
  )
