add_lldb_unittest(InterpreterTests
  TestArgs.cpp
  )

target_link_libraries(InterpreterTests
  ${PYTHON_LIBRARY}
  )
