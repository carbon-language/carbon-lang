get_filename_component(ClangReplaceLocation
  "${CMAKE_CURRENT_SOURCE_DIR}/../clang-replace/include" ABSOLUTE)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${ClangReplaceLocation}
  )

add_subdirectory(tool)
add_subdirectory(Core)
