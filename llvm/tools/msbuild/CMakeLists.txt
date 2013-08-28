if (WIN32)
  install(DIRECTORY .
    DESTINATION tools/msbuild
    FILES_MATCHING
    PATTERN "*.targets"
    PATTERN "*.props"
    PATTERN "*.bat"
    PATTERN ".svn" EXCLUDE
    )
endif()  
