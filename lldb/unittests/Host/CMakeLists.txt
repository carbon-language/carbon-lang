add_lldb_unittest(HostTests
  SocketAddressTest.cpp
  SocketTest.cpp
  SocketTestMock.cpp
  )

target_link_libraries(HostTests
  lldbBase
  ${PYTHON_LIBRARY}
  )
