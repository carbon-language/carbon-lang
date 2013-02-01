add_lld_library(lldHexagonELFTarget
  HexagonTargetHandler.cpp
  HexagonTargetInfo.cpp
  HexagonRelocationHandler.cpp
  )

target_link_libraries(lldHexagonELFTarget
  lldCore
  )
