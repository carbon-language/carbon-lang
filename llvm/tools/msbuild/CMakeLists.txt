if (WIN32)
  set(prop_file "Microsoft.Cpp.Win32.llvm.props")

  # CPack will install a registry key in this format that we wish to reference.
  set(REG_KEY "${CMAKE_PROJECT_NAME} ${CPACK_PACKAGE_VERSION}")
  configure_file(${prop_file}.in ${prop_file})
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${prop_file}" DESTINATION tools/msbuild)

  install(DIRECTORY .
    DESTINATION tools/msbuild
    FILES_MATCHING
    PATTERN "*.targets"
    PATTERN "*.bat"
    PATTERN ".svn" EXCLUDE
    )
endif()  
