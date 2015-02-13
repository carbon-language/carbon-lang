add_lldb_executable(argdumper
  argdumper.cpp
  )

target_link_libraries(argdumper liblldb)

install(TARGETS argdumper
  RUNTIME DESTINATION bin)
