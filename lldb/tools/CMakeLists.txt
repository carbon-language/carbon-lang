if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  add_subdirectory(debugserver)
endif()
add_subdirectory(driver)
