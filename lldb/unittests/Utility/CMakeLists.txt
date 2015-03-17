add_lldb_unittest(UtilityTests
  StringExtractorTest.cpp
  UriParserTest.cpp
  )

target_link_libraries(UtilityTests
  lldbBase
  ${PYTHON_LIBRARY}
  )
