set(LLDB_USED_LIBS
  lldbInterpreter
  ${PYTHON_LIBRARY}
  )

add_lldb_unittest(InterpreterTests
  TestArgs.cpp
  )
