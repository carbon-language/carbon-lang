
if (DOXYGEN_FOUND)
if (LLVM_ENABLE_DOXYGEN)
  set(abs_srcdir ${CMAKE_CURRENT_SOURCE_DIR})
  set(abs_builddir ${LLVM_BINARY_DIR})

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
      if (NOT ${Name} EQUALS clang)
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
  set(abs_srcdir)
  set(abs_builddir)
  set(DOT)

  add_custom_target(doxygen-clang
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating clang doxygen documentation." VERBATIM)

  if (LLVM_BUILD_DOCS)
    add_dependencies(doxygen doxygen-clang)
  endif()

  if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html
      DESTINATION docs/html)
  endif()
endif()
endif()
