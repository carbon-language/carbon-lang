
if (DOXYGEN_FOUND)
if (LLVM_ENABLE_DOXYGEN)
  set(abs_top_srcdir ${LLVM_MAIN_SRC_DIR})
  set(abs_top_builddir ${LLVM_BINARY_DIR})
  
  if (HAVE_DOT)
    set(DOT ${LLVM_PATH_DOT})
  endif()

  if (DOXYGEN_EXTERNAL_SEARCH)
    set(SEARCHENGINE "YES")
    set(SERVER_BASED_SEARCH "YES")
    set(EXTERNAL_SEARCH "YES")

    set(EXTRA_SEARCH_MAPPINGS "")
    foreach(NameAndValue ${DOXYGEN_SEARCH_MAPPINGS})
      # Strip leading spaces
      string(REGEX REPLACE "^[ ]+" "" NameAndValue ${NameAndValue})
      # Find variable name
      string(REGEX MATCH "^[^=]+" Name ${NameAndValue})
      # Find the value
      string(REPLACE "${Name}=" "" Value ${NameAndValue})
      # Set the variable
      if (NOT ${Name} EQUALS llvm)
	set(EXTRA_SEARCH_MAPPINGS "${EXTRA_SEARCH_MAPPINGS} ${LLVM_BINARY_DIR}/${NameAndValue}")
      endif()
    endforeach()
  else()
    set(SEARCHENGINE "NO")
    set(SERVER_BASED_SEARCH "NO")
    set(EXTERNAL_SEARCH "NO")
    set(EXTRA_SEARCH_MAPPINGS "")
  endif()
  
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg @ONLY)
  set(abs_top_srcdir)
  set(abs_top_builddir)
  set(DOT)

  add_custom_target(doxygen-llvm
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating llvm doxygen documentation." VERBATIM)

  if (LLVM_BUILD_DOCS)
    add_dependencies(doxygen doxygen-llvm)
  endif()

  if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html
      DESTINATION docs/html)
  endif()
endif()
endif()
