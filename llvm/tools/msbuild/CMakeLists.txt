if (WIN32)
  set(prop_file_in "Microsoft.Cpp.Win32.llvm.props.in")
  set(prop_file_v100 "Microsoft.Cpp.Win32.LLVM-vs2010.props")
  set(prop_file_v110 "Microsoft.Cpp.Win32.LLVM-vs2012.props")

  # CPack will install a registry key in this format that we wish to reference.
  set(REG_KEY "${CMAKE_PROJECT_NAME} ${CPACK_PACKAGE_VERSION}")

  set(VS_VERSION "v100")
  configure_file(${prop_file_in} ${prop_file_v100})
  set(VS_VERSION "v110")
  configure_file(${prop_file_in} ${prop_file_v110})

  set(REG_KEY)
  set(VS_VERSION)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${prop_file_v100}" DESTINATION tools/msbuild)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${prop_file_v110}" DESTINATION tools/msbuild)

  install(DIRECTORY .
    DESTINATION tools/msbuild
    FILES_MATCHING
    PATTERN "*.targets"
    PATTERN "*.bat"
    PATTERN ".svn" EXCLUDE
    )
endif()
