add_subdirectory(libclang)
add_subdirectory(c-index-test)
add_subdirectory(arcmt-test)
add_subdirectory(c-arcmt-test)
add_subdirectory(diagtool)
add_subdirectory(driver)
add_subdirectory(clang-check)

# We support checking out the clang-tools-extra repository into the 'extra'
# subdirectory. It contains tools developed as part of the Clang/LLVM project
# on top of the Clang tooling platform. We keep them in a separate repository
# to keep the primary Clang repository small and focused.
if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/extra AND
   EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/extra/CMakeLists.txt)
  add_subdirectory(extra)
endif()
