set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  CodeGen
  Core
  IPO
  InstCombine
  Instrumentation
  MC
  MCParser
  ObjCARCOpts
  Option
  ScalarOpts
  Support
  TransformUtils
  Vectorize
  )

option(CLANG_PLUGIN_SUPPORT "Build clang with plugin support" ON)

# Support plugins. This must be before add_clang_executable as it reads
# LLVM_NO_DEAD_STRIP.
if(CLANG_PLUGIN_SUPPORT)
  set(LLVM_NO_DEAD_STRIP 1)
endif()

add_clang_executable(clang
  driver.cpp
  cc1_main.cpp
  cc1as_main.cpp
  )

target_link_libraries(clang
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangFrontendTool
  )

if(WIN32 AND NOT CYGWIN)
  # Prevent versioning if the buildhost is targeting for Win32.
else()
  set_target_properties(clang PROPERTIES VERSION ${CLANG_EXECUTABLE_VERSION})
endif()

# Support plugins.
if(CLANG_PLUGIN_SUPPORT)
  export_executable_symbols(clang)
endif()

add_dependencies(clang clang-headers)

if(UNIX)
  set(CLANGXX_LINK_OR_COPY create_symlink)
# Create a relative symlink
  set(clang_binary "clang${CMAKE_EXECUTABLE_SUFFIX}")
else()
  set(CLANGXX_LINK_OR_COPY copy)
  set(clang_binary "${LLVM_RUNTIME_OUTPUT_INTDIR}/clang${CMAKE_EXECUTABLE_SUFFIX}")
endif()

# Create the clang++ symlink in the build directory.
set(clang_pp "${LLVM_RUNTIME_OUTPUT_INTDIR}/clang++${CMAKE_EXECUTABLE_SUFFIX}")
add_custom_command(TARGET clang POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E ${CLANGXX_LINK_OR_COPY} "${clang_binary}" "${clang_pp}"
    WORKING_DIRECTORY "${LLVM_RUNTIME_OUTPUT_INTDIR}")

set_property(DIRECTORY APPEND
  PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${clang_pp})

# Create the clang-cl symlink in the build directory.
set(clang_cl "${LLVM_RUNTIME_OUTPUT_INTDIR}/clang-cl${CMAKE_EXECUTABLE_SUFFIX}")
add_custom_command(TARGET clang POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E ${CLANGXX_LINK_OR_COPY} "${clang_binary}" "${clang_cl}"
    WORKING_DIRECTORY "${LLVM_RUNTIME_OUTPUT_INTDIR}")

set_property(DIRECTORY APPEND
  PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${clang_cl})

install(TARGETS clang
  RUNTIME DESTINATION bin
  COMPONENT clang)
add_custom_target(install-clang
  DEPENDS clang
  COMMAND "${CMAKE_COMMAND}"
          -DCMAKE_INSTALL_COMPONENT=clang
          -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")

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

if(WITH_POLLY AND LINK_POLLY_INTO_TOOLS)
  target_link_libraries(clang Polly)
  if(POLLY_LINK_LIBS)
    foreach(lib ${POLLY_LINK_LIBS})
      target_link_libraries(clang ${lib})
    endforeach(lib)
  endif(POLLY_LINK_LIBS)
endif(WITH_POLLY AND LINK_POLLY_INTO_TOOLS)
