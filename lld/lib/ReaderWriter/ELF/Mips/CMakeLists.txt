set(LLVM_LINK_COMPONENTS support)

add_lld_library(lldMipsELFTarget
  MipsCtorsOrderPass.cpp
  MipsELFFlagsMerger.cpp
  MipsLinkingContext.cpp
  MipsRelocationHandler.cpp
  MipsRelocationPass.cpp
  MipsTargetHandler.cpp
  )

target_link_libraries(lldMipsELFTarget ${cmake_2_8_12_INTERFACE}
  lldCore
  )
