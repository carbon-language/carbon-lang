set(LLVM_NO_RTTI 1)

include_directories(../../source)

add_lldb_executable(lldb-gdbserver
  lldb-gdbserver.cpp
  )

target_link_libraries(lldb-gdbserver liblldb)

set_target_properties(lldb-gdbserver PROPERTIES VERSION ${LLDB_VERSION})

install(TARGETS lldb-gdbserver
  RUNTIME DESTINATION bin)
