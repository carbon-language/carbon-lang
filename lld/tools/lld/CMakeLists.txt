add_lld_executable(lld
  lld.cpp
  )

install(TARGETS lld
  RUNTIME DESTINATION bin)
