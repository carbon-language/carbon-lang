add_lld_executable(lld
  lld.cpp
  )

target_link_libraries(lld
  lldDriver
  )

install(TARGETS lld
  RUNTIME DESTINATION bin)
