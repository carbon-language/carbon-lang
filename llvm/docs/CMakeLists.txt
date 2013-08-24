
if (LLVM_BUILD_DOCS)
  set(LLVM_DOCS_ADD_TO_ALL ALL)
endif()

if (DOXYGEN_FOUND)
if (LLVM_ENABLE_DOXYGEN)
  set(abs_top_srcdir ${LLVM_MAIN_SRC_DIR})
  set(abs_top_builddir ${LLVM_BINARY_DIR})
  
  if (HAVE_DOT)
    set(DOT ${LLVM_PATH_DOT})
  endif()
  
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg @ONLY)
  set(abs_top_srcdir)
  set(abs_top_builddir)
  set(DOT)

  add_custom_target(doxygen ${LLVM_DOCS_ADD_TO_ALL}
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating doxygen documentation." VERBATIM)

  if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html
      DESTINATION docs/html)
  endif()
endif()
endif()
