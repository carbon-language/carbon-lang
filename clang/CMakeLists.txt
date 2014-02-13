cmake_minimum_required(VERSION 2.8.8)

# If we are not building as a part of LLVM, build Clang as an
# standalone project, using LLVM as an external library:
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  project(Clang)

  # Rely on llvm-config.
  set(CONFIG_OUTPUT)
  find_program(LLVM_CONFIG "llvm-config")
  if(LLVM_CONFIG)
    message(STATUS "Found LLVM_CONFIG as ${LLVM_CONFIG}")
    set(CONFIG_COMMAND ${LLVM_CONFIG}
      "--assertion-mode"
      "--bindir"
      "--libdir"
      "--includedir"
      "--prefix"
      "--src-root")
    execute_process(
      COMMAND ${CONFIG_COMMAND}
      RESULT_VARIABLE HAD_ERROR
      OUTPUT_VARIABLE CONFIG_OUTPUT
    )
    if(NOT HAD_ERROR)
      string(REGEX REPLACE
        "[ \t]*[\r\n]+[ \t]*" ";"
        CONFIG_OUTPUT ${CONFIG_OUTPUT})
    else()
      string(REPLACE ";" " " CONFIG_COMMAND_STR "${CONFIG_COMMAND}")
      message(STATUS "${CONFIG_COMMAND_STR}")
      message(FATAL_ERROR "llvm-config failed with status ${HAD_ERROR}")
    endif()
  else()
    message(FATAL_ERROR "llvm-config not found -- ${LLVM_CONFIG}")
  endif()

  list(GET CONFIG_OUTPUT 0 ENABLE_ASSERTIONS)
  list(GET CONFIG_OUTPUT 1 TOOLS_BINARY_DIR)
  list(GET CONFIG_OUTPUT 2 LIBRARY_DIR)
  list(GET CONFIG_OUTPUT 3 INCLUDE_DIR)
  list(GET CONFIG_OUTPUT 4 LLVM_OBJ_ROOT)
  list(GET CONFIG_OUTPUT 5 MAIN_SRC_DIR)

  if(NOT MSVC_IDE)
    set(LLVM_ENABLE_ASSERTIONS ${ENABLE_ASSERTIONS}
      CACHE BOOL "Enable assertions")
    # Assertions should follow llvm-config's.
    mark_as_advanced(LLVM_ENABLE_ASSERTIONS)
  endif()

  set(LLVM_TOOLS_BINARY_DIR ${TOOLS_BINARY_DIR} CACHE PATH "Path to llvm/bin")
  set(LLVM_LIBRARY_DIR ${LIBRARY_DIR} CACHE PATH "Path to llvm/lib")
  set(LLVM_MAIN_INCLUDE_DIR ${INCLUDE_DIR} CACHE PATH "Path to llvm/include")
  set(LLVM_BINARY_DIR ${LLVM_OBJ_ROOT} CACHE PATH "Path to LLVM build tree")
  set(LLVM_MAIN_SRC_DIR ${MAIN_SRC_DIR} CACHE PATH "Path to LLVM source tree")

  find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)

  set(LLVM_CMAKE_PATH "${LLVM_BINARY_DIR}/share/llvm/cmake")
  set(LLVMCONFIG_FILE "${LLVM_CMAKE_PATH}/LLVMConfig.cmake")
  if(EXISTS ${LLVMCONFIG_FILE})
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_PATH}")
    include(${LLVMCONFIG_FILE})
  else()
    message(FATAL_ERROR "Not found: ${LLVMCONFIG_FILE}")
  endif()

  # They are used as destination of target generators.
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib)

  option(LLVM_INSTALL_TOOLCHAIN_ONLY
    "Only include toolchain files in the 'install' target." OFF)

  option(LLVM_FORCE_USE_OLD_HOST_TOOLCHAIN
    "Set to ON to force using an old, unsupported host toolchain." OFF)

  include(AddLLVM)
  include(TableGen)
  include(HandleLLVMOptions)

  set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")

  if (NOT DEFINED LLVM_INCLUDE_TESTS)
    set(LLVM_INCLUDE_TESTS ON)
  endif()

  include_directories("${LLVM_BINARY_DIR}/include" "${LLVM_MAIN_INCLUDE_DIR}")
  link_directories("${LLVM_LIBRARY_DIR}")

  set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
  set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
  set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

  if(LLVM_INCLUDE_TESTS)
    # Check prebuilt llvm/utils.
    if(EXISTS ${LLVM_TOOLS_BINARY_DIR}/FileCheck${CMAKE_EXECUTABLE_SUFFIX}
        AND EXISTS ${LLVM_TOOLS_BINARY_DIR}/count${CMAKE_EXECUTABLE_SUFFIX}
        AND EXISTS ${LLVM_TOOLS_BINARY_DIR}/not${CMAKE_EXECUTABLE_SUFFIX})
      set(LLVM_UTILS_PROVIDED ON)
    endif()

    if(EXISTS ${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py)
      set(LLVM_LIT ${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py)
      if(NOT LLVM_UTILS_PROVIDED)
        add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/FileCheck utils/FileCheck)
        add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/count utils/count)
        add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/not utils/not)
        set(LLVM_UTILS_PROVIDED ON)
        set(CLANG_TEST_DEPS FileCheck count not)
      endif()
      set(UNITTEST_DIR ${LLVM_MAIN_SRC_DIR}/utils/unittest)
      if(EXISTS ${UNITTEST_DIR}/googletest/include/gtest/gtest.h
          AND NOT EXISTS ${LLVM_LIBRARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}
          AND EXISTS ${UNITTEST_DIR}/CMakeLists.txt)
        add_subdirectory(${UNITTEST_DIR} utils/unittest)
      endif()
    else()
      # Seek installed Lit.
      find_program(LLVM_LIT "lit.py" ${LLVM_MAIN_SRC_DIR}/utils/lit
        DOC "Path to lit.py")
    endif()

    if(LLVM_LIT)
      # Define the default arguments to use with 'lit', and an option for the user
      # to override.
      set(LIT_ARGS_DEFAULT "-sv")
      if (MSVC OR XCODE)
        set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
      endif()
      set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

      # On Win32 hosts, provide an option to specify the path to the GnuWin32 tools.
      if( WIN32 AND NOT CYGWIN )
        set(LLVM_LIT_TOOLS_DIR "" CACHE PATH "Path to GnuWin32 tools")
      endif()
    else()
      set(LLVM_INCLUDE_TESTS OFF)
    endif()
  endif()

  set( CLANG_BUILT_STANDALONE 1 )

  find_package(LibXml2)
  if (LIBXML2_FOUND)
    set(CLANG_HAVE_LIBXML 1)
  endif ()
endif()

set(CLANG_RESOURCE_DIR "" CACHE STRING
  "Relative directory from the Clang binary to its resource files.")

set(C_INCLUDE_DIRS "" CACHE STRING
  "Colon separated list of directories clang will search for headers.")

set(GCC_INSTALL_PREFIX "" CACHE PATH "Directory where gcc is installed." )
set(DEFAULT_SYSROOT "" CACHE PATH
  "Default <path> to all compiler invocations for --sysroot=<path>." )

set(CLANG_VENDOR "" CACHE STRING
  "Vendor-specific text for showing with version information.")

if( CLANG_VENDOR )
  add_definitions( -DCLANG_VENDOR="${CLANG_VENDOR} " )
endif()

set(CLANG_REPOSITORY_STRING "" CACHE STRING
  "Vendor-specific text for showing the repository the source is taken from.")

if(CLANG_REPOSITORY_STRING)
  add_definitions(-DCLANG_REPOSITORY_STRING="${CLANG_REPOSITORY_STRING}")
endif()

set(CLANG_VENDOR_UTI "org.llvm.clang" CACHE STRING
  "Vendor-specific uti.")

set(CLANG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

if( CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE )
  message(FATAL_ERROR "In-source builds are not allowed. CMake would overwrite "
"the makefiles distributed with LLVM. Please create a directory and run cmake "
"from there, passing the path to this source directory as the last argument. "
"This process created the file `CMakeCache.txt' and the directory "
"`CMakeFiles'. Please delete them.")
endif()

if( NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR )
  file(GLOB_RECURSE
    tablegenned_files_on_include_dir
    "${CLANG_SOURCE_DIR}/include/clang/*.inc")
  if( tablegenned_files_on_include_dir )
    message(FATAL_ERROR "Apparently there is a previous in-source build, "
"probably as the result of running `configure' and `make' on "
"${CLANG_SOURCE_DIR}. This may cause problems. The suspicious files are:\n"
"${tablegenned_files_on_include_dir}\nPlease clean the source directory.")
  endif()
endif()

# Compute the Clang version from the LLVM version.
string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" CLANG_VERSION
  ${PACKAGE_VERSION})
message(STATUS "Clang version: ${CLANG_VERSION}")

string(REGEX REPLACE "([0-9]+)\\.[0-9]+(\\.[0-9]+)?" "\\1" CLANG_VERSION_MAJOR
  ${CLANG_VERSION})
string(REGEX REPLACE "[0-9]+\\.([0-9]+)(\\.[0-9]+)?" "\\1" CLANG_VERSION_MINOR
  ${CLANG_VERSION})
if (${CLANG_VERSION} MATCHES "[0-9]+\\.[0-9]+\\.[0-9]+")
  set(CLANG_HAS_VERSION_PATCHLEVEL 1)
  string(REGEX REPLACE "[0-9]+\\.[0-9]+\\.([0-9]+)" "\\1" CLANG_VERSION_PATCHLEVEL
    ${CLANG_VERSION})
else()
  set(CLANG_HAS_VERSION_PATCHLEVEL 0)
endif()

# Configure the Version.inc file.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/clang/Basic/Version.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/clang/Basic/Version.inc)

# Add appropriate flags for GCC
if (LLVM_COMPILER_IS_GCC_COMPATIBLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common -Woverloaded-virtual -Wcast-qual -fno-strict-aliasing")

  # Enable -pedantic for Clang even if it's not enabled for LLVM.
  if (NOT LLVM_ENABLE_PEDANTIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wno-long-long")
  endif ()

  check_cxx_compiler_flag("-Werror -Wnested-anon-types" CXX_SUPPORTS_NO_NESTED_ANON_TYPES_FLAG)
  if( CXX_SUPPORTS_NO_NESTED_ANON_TYPES_FLAG )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-nested-anon-types" )
  endif()
endif ()

configure_file(
  ${CLANG_SOURCE_DIR}/include/clang/Config/config.h.cmake
  ${CLANG_BINARY_DIR}/include/clang/Config/config.h)

include(CMakeParseArguments)

function(clang_tablegen)
  # Syntax:
  # clang_tablegen output-file [tablegen-arg ...] SOURCE source-file
  # [[TARGET cmake-target-name] [DEPENDS extra-dependency ...]]
  #
  # Generates a custom command for invoking tblgen as
  #
  # tblgen source-file -o=output-file tablegen-arg ...
  #
  # and, if cmake-target-name is provided, creates a custom target for
  # executing the custom command depending on output-file. It is
  # possible to list more files to depend after DEPENDS.

  cmake_parse_arguments(CTG "" "SOURCE;TARGET" "DEPENDS" ${ARGN})

  if( NOT CTG_SOURCE )
    message(FATAL_ERROR "SOURCE source-file required by clang_tablegen")
  endif()

  set( LLVM_TARGET_DEFINITIONS ${CTG_SOURCE} )
  tablegen(CLANG ${CTG_UNPARSED_ARGUMENTS})

  list(GET CTG_UNPARSED_ARGUMENTS 0 output_file)
  if( CTG_TARGET )
    add_custom_target( ${CTG_TARGET} DEPENDS ${output_file} ${CTG_DEPENDS} )
    set_target_properties( ${CTG_TARGET} PROPERTIES FOLDER "Clang tablegenning")
  endif()
endfunction(clang_tablegen)

macro(add_clang_library name)
  cmake_parse_arguments(ARG
    ""
    ""
    "ADDITIONAL_HEADERS"
    ${ARGN})
  set(srcs)
  if(MSVC_IDE OR XCODE)
    # Add public headers
    file(RELATIVE_PATH lib_path
      ${CLANG_SOURCE_DIR}/lib/
      ${CMAKE_CURRENT_SOURCE_DIR}
    )
    if(NOT lib_path MATCHES "^[.][.]")
      file( GLOB_RECURSE headers
        ${CLANG_SOURCE_DIR}/include/clang/${lib_path}/*.h
        ${CLANG_SOURCE_DIR}/include/clang/${lib_path}/*.def
      )
      set_source_files_properties(${headers} PROPERTIES HEADER_FILE_ONLY ON)

      file( GLOB_RECURSE tds
        ${CLANG_SOURCE_DIR}/include/clang/${lib_path}/*.td
      )
      source_group("TableGen descriptions" FILES ${tds})
      set_source_files_properties(${tds}} PROPERTIES HEADER_FILE_ONLY ON)

      if(headers OR tds)
	set(srcs ${headers} ${tds})
      endif()
    endif()
  endif(MSVC_IDE OR XCODE)
  if(srcs OR ARG_ADDITIONAL_HEADERS)
    set(srcs
      ADDITIONAL_HEADERS
      ${srcs}
      ${ARG_ADDITIONAL_HEADERS} # It may contain unparsed unknown args.
      )
  endif()
  llvm_add_library(${name} ${ARG_UNPARSED_ARGUMENTS} ${srcs})

  if(TARGET ${name})
    target_link_libraries( ${name} ${LLVM_COMMON_LIBS} )

    if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY OR ${name} STREQUAL "libclang")
      install(TARGETS ${name}
        LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
        ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
        RUNTIME DESTINATION bin)
    endif()
  else()
    # Add empty "phony" target
    add_custom_target(${name})
  endif()

  set_target_properties(${name} PROPERTIES FOLDER "Clang libraries")
endmacro(add_clang_library)

macro(add_clang_executable name)
  add_llvm_executable( ${name} ${ARGN} )
  set_target_properties(${name} PROPERTIES FOLDER "Clang executables")
endmacro(add_clang_executable)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(BEFORE
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY include/clang include/clang-c
    DESTINATION include
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "config.h" EXCLUDE
    PATTERN ".svn" EXCLUDE
    )

  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/clang
    DESTINATION include
    FILES_MATCHING
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "*.inc"
    )
endif()

install(DIRECTORY include/clang-c
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
  )

add_definitions( -D_GNU_SOURCE )

option(CLANG_ENABLE_ARCMT "Build ARCMT." ON)
option(CLANG_ENABLE_REWRITER "Build rewriter." ON)
option(CLANG_ENABLE_STATIC_ANALYZER "Build static analyzer." ON)

if (NOT CLANG_ENABLE_REWRITER AND CLANG_ENABLE_ARCMT)
  message(FATAL_ERROR "Cannot disable rewriter while enabling ARCMT")
endif()

if (NOT CLANG_ENABLE_REWRITER AND CLANG_ENABLE_STATIC_ANALYZER)
  message(FATAL_ERROR "Cannot disable rewriter while enabling static analyzer")
endif()

if (NOT CLANG_ENABLE_STATIC_ANALYZER AND CLANG_ENABLE_ARCMT)
  message(FATAL_ERROR "Cannot disable static analyzer while enabling ARCMT")
endif()

if(CLANG_ENABLE_ARCMT)
  add_definitions(-DCLANG_ENABLE_ARCMT)
endif()
if(CLANG_ENABLE_REWRITER)
  add_definitions(-DCLANG_ENABLE_REWRITER)
endif()
if(CLANG_ENABLE_STATIC_ANALYZER)
  add_definitions(-DCLANG_ENABLE_STATIC_ANALYZER)
endif()

# Clang version information
set(CLANG_EXECUTABLE_VERSION
     "${CLANG_VERSION_MAJOR}.${CLANG_VERSION_MINOR}" CACHE STRING
    "Version number that will be placed into the clang executable, in the form XX.YY")
set(LIBCLANG_LIBRARY_VERSION
     "${CLANG_VERSION_MAJOR}.${CLANG_VERSION_MINOR}" CACHE STRING
    "Version number that will be placed into the libclang library , in the form XX.YY")
mark_as_advanced(CLANG_EXECUTABLE_VERSION LIBCLANG_LIBRARY_VERSION)

add_subdirectory(utils/TableGen)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(runtime)

option(CLANG_BUILD_EXAMPLES "Build CLANG example programs by default." OFF)
if (CLANG_BUILD_EXAMPLES)
  set(ENABLE_CLANG_EXAMPLES "1")
else()
  set(ENABLE_CLANG_EXAMPLES "0")
endif()
add_subdirectory(examples)

option(CLANG_INCLUDE_TESTS
       "Generate build targets for the Clang unit tests."
       ${LLVM_INCLUDE_TESTS})

if( CLANG_INCLUDE_TESTS )
  if(EXISTS ${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include/gtest/gtest.h)
    add_subdirectory(unittests)
    list(APPEND CLANG_TEST_DEPS ClangUnitTests)
    list(APPEND CLANG_TEST_PARAMS
      clang_unit_site_config=${CMAKE_CURRENT_BINARY_DIR}/test/Unit/lit.site.cfg
      )
  endif()
  add_subdirectory(test)

  if(CLANG_BUILT_STANDALONE)
    # Add a global check rule now that all subdirectories have been traversed
    # and we know the total set of lit testsuites.
    get_property(LLVM_LIT_TESTSUITES GLOBAL PROPERTY LLVM_LIT_TESTSUITES)
    get_property(LLVM_LIT_PARAMS GLOBAL PROPERTY LLVM_LIT_PARAMS)
    get_property(LLVM_LIT_DEPENDS GLOBAL PROPERTY LLVM_LIT_DEPENDS)
    get_property(LLVM_LIT_EXTRA_ARGS GLOBAL PROPERTY LLVM_LIT_EXTRA_ARGS)
    add_lit_target(check-all
      "Running all regression tests"
      ${LLVM_LIT_TESTSUITES}
      PARAMS ${LLVM_LIT_PARAMS}
      DEPENDS ${LLVM_LIT_DEPENDS}
      ARGS ${LLVM_LIT_EXTRA_ARGS}
      )
  endif()
endif()

option(CLANG_INCLUDE_DOCS "Generate build targets for the Clang docs."
  ${LLVM_INCLUDE_DOCS})
if( CLANG_INCLUDE_DOCS )
  add_subdirectory(docs)
endif()

# Workaround for MSVS10 to avoid the Dialog Hell
# FIXME: This could be removed with future version of CMake.
if( CLANG_BUILT_STANDALONE AND MSVC_VERSION EQUAL 1600 )
  set(CLANG_SLN_FILENAME "${CMAKE_CURRENT_BINARY_DIR}/Clang.sln")
  if( EXISTS "${CLANG_SLN_FILENAME}" )
    file(APPEND "${CLANG_SLN_FILENAME}" "\n# This should be regenerated!\n")
  endif()
endif()

set(CLANG_ORDER_FILE "" CACHE FILEPATH
  "Order file to use when compiling clang in order to improve startup time.")
