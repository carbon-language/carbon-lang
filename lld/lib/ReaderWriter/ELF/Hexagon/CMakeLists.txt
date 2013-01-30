add_lld_library(lldHexagonELFTarget
  HexagonTargetHandler.cpp
  HexagonTargetInfo.cpp
  )

target_link_libraries(lldHexagonELFTarget
  lldCore
  )
