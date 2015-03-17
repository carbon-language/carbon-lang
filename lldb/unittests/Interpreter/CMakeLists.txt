add_lldb_unittest(InterpreterTests
  TestArgs.cpp
  )

target_link_libraries(InterpreterTests
  lldbInterpreter
  ${PYTHON_LIBRARY}
  )
