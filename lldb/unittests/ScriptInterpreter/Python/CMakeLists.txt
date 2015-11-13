add_lldb_unittest(ScriptInterpreterPythonTests
  PythonDataObjectsTests.cpp
  PythonTestSuite.cpp
  )

  target_link_libraries(ScriptInterpreterPythonTests lldbPluginScriptInterpreterPython ${PYTHON_LIBRARY})
  