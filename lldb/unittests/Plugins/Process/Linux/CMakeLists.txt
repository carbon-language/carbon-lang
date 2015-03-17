add_lldb_unittest(ProcessLinuxTests
  ThreadStateCoordinatorTest.cpp
  ThreadStateCoordinatorTestMock.cpp
  )

target_link_libraries(ProcessLinuxTests
  lldbBase
  ${PYTHON_LIBRARY}
  )
