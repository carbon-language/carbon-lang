set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  bitreader
  bitwriter
  irreader
  codegen
  instrumentation
  ipo
  linker
  selectiondag
  )

add_clang_executable(clang
  driver.cpp
  cc1_main.cpp
  cc1as_main.cpp
  )

target_link_libraries(clang
  clangFrontendTool
  clangAST
  clangAnalysis
  clangBasic
  clangCodeGen
  clangDriver
  clangEdit
  clangFrontend
  clangLex
  clangParse
  clangEdit
  clangSema
  clangSerialization
  )

if(CLANG_ENABLE_STATIC_ANALYZER)
  target_link_libraries(clang
    clangStaticAnalyzerFrontend
    clangStaticAnalyzerCheckers
    clangStaticAnalyzerCore
    )
endif()

if(CLANG_ENABLE_ARCMT)
  target_link_libraries(clang
    clangARCMigrate
    )
endif()

if(CLANG_ENABLE_REWRITER)
  target_link_libraries(clang
    clangRewriteCore
    clangRewriteFrontend
    )
endif()

set_target_properties(clang PROPERTIES VERSION ${CLANG_EXECUTABLE_VERSION})
set_target_properties(clang PROPERTIES ENABLE_EXPORTS 1)

add_dependencies(clang clang-headers)

if(UNIX)
  set(CLANGXX_LINK_OR_COPY create_symlink)
# Create a relative symlink
  set(clang_binary "clang${CMAKE_EXECUTABLE_SUFFIX}")
else()
  set(CLANGXX_LINK_OR_COPY copy)
  set(clang_binary "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/clang${CMAKE_EXECUTABLE_SUFFIX}")
endif()

# Create the clang++ symlink in the build directory.
set(clang_pp "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/clang++${CMAKE_EXECUTABLE_SUFFIX}")
add_custom_command(TARGET clang POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E ${CLANGXX_LINK_OR_COPY} "${clang_binary}" "${clang_pp}")

set_property(DIRECTORY APPEND
  PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${clang_pp})

# Create the clang-cl symlink in the build directory.
set(clang_cl "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/clang-cl${CMAKE_EXECUTABLE_SUFFIX}")
add_custom_command(TARGET clang POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E ${CLANGXX_LINK_OR_COPY} "${clang_binary}" "${clang_cl}")

set_property(DIRECTORY APPEND
  PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${clang_cl})

install(TARGETS clang
  RUNTIME DESTINATION bin)

# Create the clang++ and clang-cl symlinks at installation time.
install(SCRIPT clang_symlink.cmake -DCMAKE_INSTALL_PREFIX=\"${CMAKE_INSTALL_PREFIX}\")

# Configure plist creation for OS X.
set (TOOL_INFO_PLIST "Info.plist" CACHE STRING "Plist name")
if (APPLE)  
  if (CLANG_VENDOR)
    set(TOOL_INFO_NAME "${CLANG_VENDOR} clang")
  else()
    set(TOOL_INFO_NAME "clang")
  endif()

  set(TOOL_INFO_UTI "${CLANG_VENDOR_UTI}")
  set(TOOL_INFO_VERSION "${CLANG_VERSION}")
  if (LLVM_SUBMIT_VERSION)
    set(TOOL_INFO_BUILD_VERSION
      "${LLVM_SUBMIT_VERSION}.${LLVM_SUBMIT_SUBVERSION}")
  endif()
  
  set(TOOL_INFO_PLIST_OUT "${CMAKE_CURRENT_BINARY_DIR}/${TOOL_INFO_PLIST}")
  target_link_libraries(clang
    "-Wl,-sectcreate,__TEXT,__info_plist,${TOOL_INFO_PLIST_OUT}")
  configure_file("${TOOL_INFO_PLIST}.in" "${TOOL_INFO_PLIST_OUT}" @ONLY)
  
  set(TOOL_INFO_UTI)
  set(TOOL_INFO_NAME)
  set(TOOL_INFO_VERSION)
  set(TOOL_INFO_BUILD_VERSION)
endif()

if(CLANG_ORDER_FILE)
  target_link_libraries(clang "-Wl,-order_file,${CLANG_ORDER_FILE}")
endif()

