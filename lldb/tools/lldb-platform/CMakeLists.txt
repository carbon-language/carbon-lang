set(LLVM_NO_RTTI 1)

include_directories(../../source)

add_lldb_executable(lldb-platform
  lldb-platform.cpp
  )

target_link_libraries(lldb-platform liblldb)

set_target_properties(lldb-platform PROPERTIES VERSION ${LLDB_VERSION})

install(TARGETS lldb-platform
  RUNTIME DESTINATION bin)
