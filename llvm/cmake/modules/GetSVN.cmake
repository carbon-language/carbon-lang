# CMake project that writes Subversion revision information to a header.
#
# Input variables:
#   FIRST_SOURCE_DIR  - First source directory
#   FIRST_NAME        - The macro prefix for the first repository's info
#   SECOND_SOURCE_DIR - Second source directory (opt)
#   SECOND_NAME       - The macro prefix for the second repository's info (opt)
#   HEADER_FILE       - The header file to write
#
# The output header will contain macros FIRST_REPOSITORY and FIRST_REVISION,
# and SECOND_REPOSITORY and SECOND_REVISION if requested, where "FIRST" and
# "SECOND" are substituted with the names specified in the input variables.

# Chop off cmake/modules/GetSVN.cmake 
get_filename_component(LLVM_DIR "${CMAKE_SCRIPT_MODE_FILE}" PATH)
get_filename_component(LLVM_DIR "${LLVM_DIR}" PATH)
get_filename_component(LLVM_DIR "${LLVM_DIR}" PATH)

# Handle strange terminals
set(ENV{TERM} "dumb")

function(append_info name path)
  execute_process(COMMAND "${LLVM_DIR}/utils/GetSourceVersion" "${path}"
    OUTPUT_VARIABLE revision)
  string(STRIP "${revision}" revision)
  execute_process(COMMAND "${LLVM_DIR}/utils/GetRepositoryPath" "${path}"
    OUTPUT_VARIABLE repository
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(STRIP "${repository}" repository)
  file(APPEND "${HEADER_FILE}.txt"
    "#define ${name}_REVISION \"${revision}\"\n")
  file(APPEND "${HEADER_FILE}.txt"
    "#define ${name}_REPOSITORY \"${repository}\"\n")
endfunction()

append_info(${FIRST_NAME} "${FIRST_SOURCE_DIR}")
if(DEFINED SECOND_SOURCE_DIR)
  append_info(${SECOND_NAME} "${SECOND_SOURCE_DIR}")
endif()

# Copy the file only if it has changed.
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${HEADER_FILE}.txt" "${HEADER_FILE}")
file(REMOVE "${HEADER_FILE}.txt")

