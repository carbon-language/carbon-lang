get_filename_component(ClangReplaceLocation
  "${CMAKE_CURRENT_SOURCE_DIR}/../clang-apply-replacements/include" ABSOLUTE)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${ClangReplaceLocation}
  )

add_subdirectory(tool)
add_subdirectory(Core)
