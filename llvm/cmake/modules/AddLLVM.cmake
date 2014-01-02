include(LLVMParseArguments)
include(LLVMProcessSources)
include(LLVM-Config)

function(add_llvm_symbol_exports target_name export_file)
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(native_export_file "${target_name}.exports")
    add_custom_command(OUTPUT ${native_export_file}
      COMMAND sed -e "s/^/_/" < ${export_file} > ${native_export_file}
      DEPENDS ${export_file}
      VERBATIM
      COMMENT "Creating export file for ${target_name}")
    set_property(TARGET ${target_name} APPEND_STRING PROPERTY
                 LINK_FLAGS " -Wl,-exported_symbols_list,${CMAKE_CURRENT_BINARY_DIR}/${native_export_file}")
  elseif(LLVM_HAVE_LINK_VERSION_SCRIPT)
    # Gold and BFD ld require a version script rather than a plain list.
    set(native_export_file "${target_name}.exports")
    # FIXME: Don't write the "local:" line on OpenBSD.
    add_custom_command(OUTPUT ${native_export_file}
      COMMAND echo "{" > ${native_export_file}
      COMMAND grep -q "[[:alnum:]]" ${export_file} && echo "  global:" >> ${native_export_file} || :
      COMMAND sed -e "s/$/;/" -e "s/^/    /" < ${export_file} >> ${native_export_file}
      COMMAND echo "  local: *;" >> ${native_export_file}
      COMMAND echo "};" >> ${native_export_file}
      DEPENDS ${export_file}
      VERBATIM
      COMMENT "Creating export file for ${target_name}")
    set_property(TARGET ${target_name} APPEND_STRING PROPERTY
                 LINK_FLAGS "  -Wl,--version-script,${CMAKE_CURRENT_BINARY_DIR}/${native_export_file}")
  else()
    set(native_export_file "${target_name}.def")

    set(CAT "type")
    if(CYGWIN)
      set(CAT "cat")
    endif()

    # Using ${export_file} in add_custom_command directly confuses cmd.exe.
    file(TO_NATIVE_PATH ${export_file} export_file_backslashes)

    add_custom_command(OUTPUT ${native_export_file}
      COMMAND ${CMAKE_COMMAND} -E echo "EXPORTS" > ${native_export_file}
      COMMAND ${CAT} ${export_file_backslashes} >> ${native_export_file}
      DEPENDS ${export_file}
      VERBATIM
      COMMENT "Creating export file for ${target_name}")
    if(CYGWIN OR MINGW)
      set_property(TARGET ${target_name} APPEND_STRING PROPERTY
                   LINK_FLAGS " ${CMAKE_CURRENT_BINARY_DIR}/${native_export_file}")
    else()
      set_property(TARGET ${target_name} APPEND_STRING PROPERTY
                   LINK_FLAGS " /DEF:${CMAKE_CURRENT_BINARY_DIR}/${native_export_file}")
    endif()
  endif()

  add_custom_target(${target_name}_exports DEPENDS ${native_export_file})

  get_property(srcs TARGET ${target_name} PROPERTY SOURCES)
  foreach(src ${srcs})
    get_filename_component(extension ${src} EXT)
    if(extension STREQUAL ".cpp")
      set(first_source_file ${src})
      break()
    endif()
  endforeach()

  # Force re-linking when the exports file changes. Actually, it
  # forces recompilation of the source file. The LINK_DEPENDS target
  # property only works for makefile-based generators.
  set_property(SOURCE ${first_source_file} APPEND PROPERTY
    OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${native_export_file})

  set_property(DIRECTORY APPEND
    PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${native_export_file})

  add_dependencies(${target_name} ${target_name}_exports)
endfunction(add_llvm_symbol_exports)

function(add_dead_strip target_name)
  if(NOT CYGWIN AND NOT MINGW AND NOT MSVC)
    if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
       SET(CMAKE_CXX_FLAGS
           "${CMAKE_CXX_FLAGS}  -ffunction-sections -fdata-sections"
           PARENT_SCOPE)
    endif()
  endif()
  if(NOT LLVM_NO_DEAD_STRIP)
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
      set_property(TARGET ${target_name} APPEND_STRING PROPERTY
                   LINK_FLAGS " -Wl,-dead_strip")
    elseif(NOT WIN32)
      set_property(TARGET ${target_name} APPEND_STRING PROPERTY
                   LINK_FLAGS " -Wl,--gc-sections")
    endif()
  endif()
endfunction(add_dead_strip)

# Set each output directory according to ${CMAKE_CONFIGURATION_TYPES}.
# Note: Don't set variables CMAKE_*_OUTPUT_DIRECTORY any more,
# or a certain builder, for eaxample, msbuild.exe, would be confused.
function(set_output_directory target bindir libdir)
  if(NOT "${CMAKE_CFG_INTDIR}" STREQUAL ".")
    foreach(build_mode ${CMAKE_CONFIGURATION_TYPES})
      string(TOUPPER "${build_mode}" CONFIG_SUFFIX)
      string(REPLACE ${CMAKE_CFG_INTDIR} ${build_mode} bi ${bindir})
      string(REPLACE ${CMAKE_CFG_INTDIR} ${build_mode} li ${libdir})
      set_target_properties(${target} PROPERTIES "RUNTIME_OUTPUT_DIRECTORY_${CONFIG_SUFFIX}" ${bi})
      set_target_properties(${target} PROPERTIES "ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_SUFFIX}" ${li})
      set_target_properties(${target} PROPERTIES "LIBRARY_OUTPUT_DIRECTORY_${CONFIG_SUFFIX}" ${li})
    endforeach()
  else()
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${bindir})
    set_target_properties(${target} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${libdir})
    set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${libdir})
  endif()
endfunction()

macro(add_llvm_library name)
  llvm_process_sources( ALL_FILES ${ARGN} )
  add_library( ${name} ${ALL_FILES} )
  set_output_directory(${name} ${LLVM_RUNTIME_OUTPUT_INTDIR} ${LLVM_LIBRARY_OUTPUT_INTDIR})
  set_property( GLOBAL APPEND PROPERTY LLVM_LIBS ${name} )
  add_dead_strip( ${name} )
  if( LLVM_COMMON_DEPENDS )
    add_dependencies( ${name} ${LLVM_COMMON_DEPENDS} )
  endif( LLVM_COMMON_DEPENDS )

  if( BUILD_SHARED_LIBS )
    llvm_config( ${name} ${LLVM_LINK_COMPONENTS} )
    if (MSVC)
      set_target_properties(${name}
        PROPERTIES
        IMPORT_SUFFIX ".imp")
    endif ()

    if (LLVM_EXPORTED_SYMBOL_FILE)
      add_llvm_symbol_exports( ${name} ${LLVM_EXPORTED_SYMBOL_FILE} )
    endif()
  endif()

  # Ensure that the system libraries always comes last on the
  # list. Without this, linking the unit tests on MinGW fails.
  link_system_libs( ${name} )

  if( EXCLUDE_FROM_ALL )
    set_target_properties( ${name} PROPERTIES EXCLUDE_FROM_ALL ON)
  else()
    if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY OR ${name} STREQUAL "LTO")
      install(TARGETS ${name}
        LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
        ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX})
    endif()
  endif()
  set_target_properties(${name} PROPERTIES FOLDER "Libraries")

  # Add the explicit dependency information for this library.
  #
  # It would be nice to verify that we have the dependencies for this library
  # name, but using get_property(... SET) doesn't suffice to determine if a
  # property has been set to an empty value.
  get_property(lib_deps GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_${name})
  target_link_libraries(${name} ${lib_deps})
endmacro(add_llvm_library name)

macro(add_llvm_loadable_module name)
  if( NOT LLVM_ON_UNIX OR CYGWIN )
    message(STATUS "Loadable modules not supported on this platform.
${name} ignored.")
    # Add empty "phony" target
    add_custom_target(${name})
  else()
    llvm_process_sources( ALL_FILES ${ARGN} )
    if (MODULE)
      set(libkind MODULE)
    else()
      set(libkind SHARED)
    endif()

    add_library( ${name} ${libkind} ${ALL_FILES} )
    set_output_directory(${name} ${LLVM_RUNTIME_OUTPUT_INTDIR} ${LLVM_LIBRARY_OUTPUT_INTDIR})
    set_target_properties( ${name} PROPERTIES PREFIX "" )
    add_dead_strip( ${name} )

    if (LLVM_EXPORTED_SYMBOL_FILE)
      add_llvm_symbol_exports( ${name} ${LLVM_EXPORTED_SYMBOL_FILE} )
    endif(LLVM_EXPORTED_SYMBOL_FILE)

    llvm_config( ${name} ${LLVM_LINK_COMPONENTS} )
    link_system_libs( ${name} )

    if (APPLE)
      # Darwin-specific linker flags for loadable modules.
      set_property(TARGET ${name} APPEND_STRING PROPERTY
        LINK_FLAGS " -Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
    endif()

    if( EXCLUDE_FROM_ALL )
      set_target_properties( ${name} PROPERTIES EXCLUDE_FROM_ALL ON)
    else()
      if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
        install(TARGETS ${name}
          LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
          ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX})
      endif()
    endif()
  endif()

  set_target_properties(${name} PROPERTIES FOLDER "Loadable modules")
endmacro(add_llvm_loadable_module name)


macro(add_llvm_executable name)
  llvm_process_sources( ALL_FILES ${ARGN} )
  if( EXCLUDE_FROM_ALL )
    add_executable(${name} EXCLUDE_FROM_ALL ${ALL_FILES})
  else()
    add_executable(${name} ${ALL_FILES})
  endif()
  add_dead_strip( ${name} )

  if (LLVM_EXPORTED_SYMBOL_FILE)
    add_llvm_symbol_exports( ${name} ${LLVM_EXPORTED_SYMBOL_FILE} )
  endif(LLVM_EXPORTED_SYMBOL_FILE)

  set(EXCLUDE_FROM_ALL OFF)
  set_output_directory(${name} ${LLVM_RUNTIME_OUTPUT_INTDIR} ${LLVM_LIBRARY_OUTPUT_INTDIR})
  llvm_config( ${name} ${LLVM_LINK_COMPONENTS} )
  if( LLVM_COMMON_DEPENDS )
    add_dependencies( ${name} ${LLVM_COMMON_DEPENDS} )
  endif( LLVM_COMMON_DEPENDS )
  link_system_libs( ${name} )
endmacro(add_llvm_executable name)


set (LLVM_TOOLCHAIN_TOOLS
  llvm-ar
  llvm-objdump
  )

macro(add_llvm_tool name)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LLVM_TOOLS_BINARY_DIR})
  if( NOT LLVM_BUILD_TOOLS )
    set(EXCLUDE_FROM_ALL ON)
  endif()
  add_llvm_executable(${name} ${ARGN})

  list(FIND LLVM_TOOLCHAIN_TOOLS ${name} LLVM_IS_${name}_TOOLCHAIN_TOOL)
  if (LLVM_IS_${name}_TOOLCHAIN_TOOL GREATER -1 OR NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
    if( LLVM_BUILD_TOOLS )
      install(TARGETS ${name} RUNTIME DESTINATION bin)
    endif()
  endif()
  set_target_properties(${name} PROPERTIES FOLDER "Tools")
endmacro(add_llvm_tool name)


macro(add_llvm_example name)
#  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LLVM_EXAMPLES_BINARY_DIR})
  if( NOT LLVM_BUILD_EXAMPLES )
    set(EXCLUDE_FROM_ALL ON)
  endif()
  add_llvm_executable(${name} ${ARGN})
  if( LLVM_BUILD_EXAMPLES )
    install(TARGETS ${name} RUNTIME DESTINATION examples)
  endif()
  set_target_properties(${name} PROPERTIES FOLDER "Examples")
endmacro(add_llvm_example name)


macro(add_llvm_utility name)
  add_llvm_executable(${name} ${ARGN})
  set_target_properties(${name} PROPERTIES FOLDER "Utils")
endmacro(add_llvm_utility name)


macro(add_llvm_target target_name)
  include_directories(BEFORE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})
  add_llvm_library(LLVM${target_name} ${ARGN} ${TABLEGEN_OUTPUT})
  set( CURRENT_LLVM_TARGET LLVM${target_name} )
endmacro(add_llvm_target)

# Add external project that may want to be built as part of llvm such as Clang,
# lld, and Polly. This adds two options. One for the source directory of the
# project, which defaults to ${CMAKE_CURRENT_SOURCE_DIR}/${name}. Another to
# enable or disable building it with everything else.
# Additional parameter can be specified as the name of directory.
macro(add_llvm_external_project name)
  set(add_llvm_external_dir "${ARGN}")
  if("${add_llvm_external_dir}" STREQUAL "")
    set(add_llvm_external_dir ${name})
  endif()
  list(APPEND LLVM_IMPLICIT_PROJECT_IGNORE "${CMAKE_CURRENT_SOURCE_DIR}/${add_llvm_external_dir}")
  string(REPLACE "-" "_" nameUNDERSCORE ${name})
  string(TOUPPER ${nameUNDERSCORE} nameUPPER)
  set(LLVM_EXTERNAL_${nameUPPER}_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${add_llvm_external_dir}"
      CACHE PATH "Path to ${name} source directory")
  if (NOT ${LLVM_EXTERNAL_${nameUPPER}_SOURCE_DIR} STREQUAL ""
      AND EXISTS ${LLVM_EXTERNAL_${nameUPPER}_SOURCE_DIR}/CMakeLists.txt)
    option(LLVM_EXTERNAL_${nameUPPER}_BUILD
           "Whether to build ${name} as part of LLVM" ON)
    if (LLVM_EXTERNAL_${nameUPPER}_BUILD)
      add_subdirectory(${LLVM_EXTERNAL_${nameUPPER}_SOURCE_DIR} ${add_llvm_external_dir})
    endif()
  endif()
endmacro(add_llvm_external_project)

macro(add_llvm_tool_subdirectory name)
  list(APPEND LLVM_IMPLICIT_PROJECT_IGNORE "${CMAKE_CURRENT_SOURCE_DIR}/${name}")
  add_subdirectory(${name})
endmacro(add_llvm_tool_subdirectory)

macro(ignore_llvm_tool_subdirectory name)
  list(APPEND LLVM_IMPLICIT_PROJECT_IGNORE "${CMAKE_CURRENT_SOURCE_DIR}/${name}")
endmacro(ignore_llvm_tool_subdirectory)

function(add_llvm_implicit_external_projects)
  set(list_of_implicit_subdirs "")
  file(GLOB sub-dirs "${CMAKE_CURRENT_SOURCE_DIR}/*")
  foreach(dir ${sub-dirs})
    if(IS_DIRECTORY "${dir}")
      list(FIND LLVM_IMPLICIT_PROJECT_IGNORE "${dir}" tool_subdir_ignore)
      if( tool_subdir_ignore EQUAL -1
          AND EXISTS "${dir}/CMakeLists.txt")
        get_filename_component(fn "${dir}" NAME)
        list(APPEND list_of_implicit_subdirs "${fn}")
      endif()
    endif()
  endforeach()

  foreach(external_proj ${list_of_implicit_subdirs})
    add_llvm_external_project("${external_proj}")
  endforeach()
endfunction(add_llvm_implicit_external_projects)

# Generic support for adding a unittest.
function(add_unittest test_suite test_name)
  if( NOT LLVM_BUILD_TESTS )
    set(EXCLUDE_FROM_ALL ON)
  endif()

  add_llvm_executable(${test_name} ${ARGN})
  set(outdir ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR})
  set_output_directory(${test_name} ${outdir} ${outdir})
  target_link_libraries(${test_name}
    gtest
    gtest_main
    LLVMSupport # gtest needs it for raw_ostream.
    )

  add_dependencies(${test_suite} ${test_name})
  get_target_property(test_suite_folder ${test_suite} FOLDER)
  if (NOT ${test_suite_folder} STREQUAL "NOTFOUND")
    set_property(TARGET ${test_name} PROPERTY FOLDER "${test_suite_folder}")
  endif ()

  # Visual Studio 2012 only supports up to 8 template parameters in
  # std::tr1::tuple by default, but gtest requires 10
  if (MSVC AND MSVC_VERSION EQUAL 1700)
    set_property(TARGET ${test_name} APPEND PROPERTY COMPILE_DEFINITIONS _VARIADIC_MAX=10)
  endif ()

  include_directories(${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include)
  set_property(TARGET ${test_name} APPEND PROPERTY COMPILE_DEFINITIONS GTEST_HAS_RTTI=0)
  if (NOT LLVM_ENABLE_THREADS)
    set_property(TARGET ${test_name} APPEND PROPERTY COMPILE_DEFINITIONS GTEST_HAS_PTHREAD=0)
  endif ()

  get_property(target_compile_flags TARGET ${test_name} PROPERTY COMPILE_FLAGS)
  if (LLVM_COMPILER_IS_GCC_COMPATIBLE)
    set(target_compile_flags "${target_compile_flags} -fno-rtti")
  elseif (MSVC)
    llvm_replace_compiler_option(target_compile_flags "/GR" "/GR-")
  endif ()

  if (SUPPORTS_NO_VARIADIC_MACROS_FLAG)
    set(target_compile_flags "${target_compile_flags} -Wno-variadic-macros")
  endif ()
  set_property(TARGET ${test_name} PROPERTY COMPILE_FLAGS "${target_compile_flags}")
endfunction()

# This function provides an automatic way to 'configure'-like generate a file
# based on a set of common and custom variables, specifically targeting the
# variables needed for the 'lit.site.cfg' files. This function bundles the
# common variables that any Lit instance is likely to need, and custom
# variables can be passed in.
function(configure_lit_site_cfg input output)
  foreach(c ${LLVM_TARGETS_TO_BUILD})
    set(TARGETS_BUILT "${TARGETS_BUILT} ${c}")
  endforeach(c)
  set(TARGETS_TO_BUILD ${TARGETS_BUILT})

  set(SHLIBEXT "${LTDL_SHLIB_EXT}")

  if(BUILD_SHARED_LIBS)
    set(LLVM_SHARED_LIBS_ENABLED "1")
  else()
    set(LLVM_SHARED_LIBS_ENABLED "0")
  endif(BUILD_SHARED_LIBS)

  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(SHLIBPATH_VAR "DYLD_LIBRARY_PATH")
  else() # Default for all other unix like systems.
    # CMake hardcodes the library locaction using rpath.
    # Therefore LD_LIBRARY_PATH is not required to run binaries in the
    # build dir. We pass it anyways.
    set(SHLIBPATH_VAR "LD_LIBRARY_PATH")
  endif()

  # Configuration-time: See Unit/lit.site.cfg.in
  if (CMAKE_CFG_INTDIR STREQUAL ".")
    set(LLVM_BUILD_MODE ".")
  else ()
    set(LLVM_BUILD_MODE "%(build_mode)s")
  endif ()

  set(LLVM_SOURCE_DIR ${LLVM_MAIN_SRC_DIR})
  set(LLVM_BINARY_DIR ${LLVM_BINARY_DIR})
  string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} LLVM_TOOLS_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})
  string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} LLVM_LIBS_DIR  ${LLVM_LIBRARY_OUTPUT_INTDIR})
  string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} SHLIBDIR  ${LLVM_LIBRARY_OUTPUT_INTDIR})
  set(PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE})
  set(ENABLE_SHARED ${LLVM_SHARED_LIBS_ENABLED})
  set(SHLIBPATH_VAR ${SHLIBPATH_VAR})

  if(LLVM_ENABLE_ASSERTIONS AND NOT MSVC_IDE)
    set(ENABLE_ASSERTIONS "1")
  else()
    set(ENABLE_ASSERTIONS "0")
  endif()

  set(HOST_OS ${CMAKE_SYSTEM_NAME})
  set(HOST_ARCH ${CMAKE_SYSTEM_PROCESSOR})

  if (CLANG_ENABLE_ARCMT)
    set(ENABLE_CLANG_ARCMT "1")
  else()
    set(ENABLE_CLANG_ARCMT "0")
  endif()
  if (CLANG_ENABLE_REWRITER)
    set(ENABLE_CLANG_REWRITER "1")
  else()
    set(ENABLE_CLANG_REWRITER "0")
  endif()
  if (CLANG_ENABLE_STATIC_ANALYZER)
    set(ENABLE_CLANG_STATIC_ANALYZER "1")
  else()
    set(ENABLE_CLANG_STATIC_ANALYZER "0")
  endif()

  configure_file(${input} ${output} @ONLY)
endfunction()

# A raw function to create a lit target. This is used to implement the testuite
# management functions.
function(add_lit_target target comment)
  parse_arguments(ARG "PARAMS;DEPENDS;ARGS" "" ${ARGN})
  set(LIT_ARGS "${ARG_ARGS} ${LLVM_LIT_ARGS}")
  separate_arguments(LIT_ARGS)
  if (NOT CMAKE_CFG_INTDIR STREQUAL ".")
    list(APPEND LIT_ARGS --param build_mode=${CMAKE_CFG_INTDIR})
  endif ()
  set(LIT_COMMAND
    ${PYTHON_EXECUTABLE}
    ${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py
    ${LIT_ARGS}
    )
  foreach(param ${ARG_PARAMS})
    list(APPEND LIT_COMMAND --param ${param})
  endforeach()
  if( ARG_DEPENDS )
    add_custom_target(${target}
      COMMAND ${LIT_COMMAND} ${ARG_DEFAULT_ARGS}
      COMMENT "${comment}"
      )
    add_dependencies(${target} ${ARG_DEPENDS})
  else()
    add_custom_target(${target}
      COMMAND ${CMAKE_COMMAND} -E echo "${target} does nothing, no tools built.")
    message(STATUS "${target} does nothing.")
  endif()

  # Tests should be excluded from "Build Solution".
  set_target_properties(${target} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD ON)
endfunction()

# A function to add a set of lit test suites to be driven through 'check-*' targets.
function(add_lit_testsuite target comment)
  parse_arguments(ARG "PARAMS;DEPENDS;ARGS" "" ${ARGN})

  # EXCLUDE_FROM_ALL excludes the test ${target} out of check-all.
  if(NOT EXCLUDE_FROM_ALL)
    # Register the testsuites, params and depends for the global check rule.
    set_property(GLOBAL APPEND PROPERTY LLVM_LIT_TESTSUITES ${ARG_DEFAULT_ARGS})
    set_property(GLOBAL APPEND PROPERTY LLVM_LIT_PARAMS ${ARG_PARAMS})
    set_property(GLOBAL APPEND PROPERTY LLVM_LIT_DEPENDS ${ARG_DEPENDS})
    set_property(GLOBAL APPEND PROPERTY LLVM_LIT_EXTRA_ARGS ${ARG_ARGS})
  endif()

  # Produce a specific suffixed check rule.
  add_lit_target(${target} ${comment}
    ${ARG_DEFAULT_ARGS}
    PARAMS ${ARG_PARAMS}
    DEPENDS ${ARG_DEPENDS}
    ARGS ${ARG_ARGS}
    )
endfunction()
