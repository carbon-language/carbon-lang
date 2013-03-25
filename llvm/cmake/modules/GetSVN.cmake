# CMake project that writes Subversion revision information to a header.
#
# Input variables:
#   FIRST_SOURCE_DIR  - First source directory
#   FIRST_REPOSITORY  - The macro to define to the first revision number.
#   SECOND_SOURCE_DIR - Second source directory
#   SECOND_REPOSITORY - The macro to define to the second revision number.
#   HEADER_FILE       - The header file to write
include(FindSubversion)
if (Subversion_FOUND AND EXISTS "${FIRST_SOURCE_DIR}/.svn")
  # Repository information for the first repository.
  Subversion_WC_INFO(${FIRST_SOURCE_DIR} MY)
  file(WRITE ${HEADER_FILE}.txt "#define ${FIRST_REPOSITORY} \"${MY_WC_REVISION}\"\n")

  # Repository information for the second repository.
  if (EXISTS "${SECOND_SOURCE_DIR}/.svn")
    Subversion_WC_INFO(${SECOND_SOURCE_DIR} MY)
    file(APPEND ${HEADER_FILE}.txt 
      "#define ${SECOND_REPOSITORY} \"${MY_WC_REVISION}\"\n")
  endif ()

  # Copy the file only if it has changed.
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${HEADER_FILE}.txt ${HEADER_FILE})
endif()
