set(LLVM_USED_LIBS
  lldCore
  )

set(LLVM_LINK_COMPONENTS
  support
  )

add_lld_executable(lld-core
  lld-core.cpp
  )

install(TARGETS lld-core
  RUNTIME DESTINATION bin)
