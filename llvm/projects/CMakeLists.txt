# Discover the projects that use CMake in the subdirectories.
# Note that explicit cmake invocation is required every time a new project is
# added or removed.
file(GLOB entries *)
foreach(entry ${entries})
  if(IS_DIRECTORY ${entry} AND EXISTS ${entry}/CMakeLists.txt)
    if(NOT ${entry} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR}/compiler-rt)
      add_subdirectory(${entry})
    endif()
  endif()
endforeach(entry)
