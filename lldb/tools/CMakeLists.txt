if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  add_subdirectory(debugserver)
endif()
if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
  add_subdirectory(driver)
endif()

