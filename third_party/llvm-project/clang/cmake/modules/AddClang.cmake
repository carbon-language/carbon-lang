include(LLVMDistributionSupport)

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

  cmake_parse_arguments(CTG "" "SOURCE;TARGET" "" ${ARGN})

  if( NOT CTG_SOURCE )
    message(FATAL_ERROR "SOURCE source-file required by clang_tablegen")
  endif()

  set( CLANG_TABLEGEN_ARGUMENTS "" )
  set( LLVM_TARGET_DEFINITIONS ${CTG_SOURCE} )
  tablegen(CLANG ${CTG_UNPARSED_ARGUMENTS} ${CLANG_TABLEGEN_ARGUMENTS})

  if(CTG_TARGET)
    add_public_tablegen_target(${CTG_TARGET})
    set_target_properties( ${CTG_TARGET} PROPERTIES FOLDER "Clang tablegenning")
    set_property(GLOBAL APPEND PROPERTY CLANG_TABLEGEN_TARGETS ${CTG_TARGET})
  endif()
endfunction(clang_tablegen)

macro(set_clang_windows_version_resource_properties name)
  if(DEFINED windows_resource_file)
    set_windows_version_resource_properties(${name} ${windows_resource_file}
      VERSION_MAJOR ${CLANG_VERSION_MAJOR}
      VERSION_MINOR ${CLANG_VERSION_MINOR}
      VERSION_PATCHLEVEL ${CLANG_VERSION_PATCHLEVEL}
      VERSION_STRING "${CLANG_VERSION} (${BACKEND_PACKAGE_STRING})"
      PRODUCT_NAME "clang")
  endif()
endmacro()

macro(add_clang_subdirectory name)
  add_llvm_subdirectory(CLANG TOOL ${name})
endmacro()

macro(add_clang_library name)
  cmake_parse_arguments(ARG
    "SHARED;STATIC;INSTALL_WITH_TOOLCHAIN"
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

  if(ARG_SHARED AND ARG_STATIC)
    set(LIBTYPE SHARED STATIC)
  elseif(ARG_SHARED)
    set(LIBTYPE SHARED)
  else()
    # llvm_add_library ignores BUILD_SHARED_LIBS if STATIC is explicitly set,
    # so we need to handle it here.
    if(BUILD_SHARED_LIBS)
      set(LIBTYPE SHARED)
    else()
      set(LIBTYPE STATIC)
    endif()
    if(NOT XCODE)
      # The Xcode generator doesn't handle object libraries correctly.
      list(APPEND LIBTYPE OBJECT)
    endif()
    set_property(GLOBAL APPEND PROPERTY CLANG_STATIC_LIBS ${name})
  endif()
  llvm_add_library(${name} ${LIBTYPE} ${ARG_UNPARSED_ARGUMENTS} ${srcs})

  set(libs ${name})
  if(ARG_SHARED AND ARG_STATIC)
    list(APPEND libs ${name}_static)
  endif()

  foreach(lib ${libs})
    if(TARGET ${lib})
      target_link_libraries(${lib} INTERFACE ${LLVM_COMMON_LIBS})

      if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY OR ARG_INSTALL_WITH_TOOLCHAIN)
        get_target_export_arg(${name} Clang export_to_clangtargets UMBRELLA clang-libraries)
        install(TARGETS ${lib}
          COMPONENT ${lib}
          ${export_to_clangtargets}
          LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
          ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
          RUNTIME DESTINATION bin)

        if (NOT LLVM_ENABLE_IDE)
          add_llvm_install_targets(install-${lib}
                                   DEPENDS ${lib}
                                   COMPONENT ${lib})
        endif()

        set_property(GLOBAL APPEND PROPERTY CLANG_LIBS ${lib})
      endif()
      set_property(GLOBAL APPEND PROPERTY CLANG_EXPORTS ${lib})
    else()
      # Add empty "phony" target
      add_custom_target(${lib})
    endif()
  endforeach()

  set_target_properties(${name} PROPERTIES FOLDER "Clang libraries")
  set_clang_windows_version_resource_properties(${name})
endmacro(add_clang_library)

macro(add_clang_executable name)
  add_llvm_executable( ${name} ${ARGN} )
  set_target_properties(${name} PROPERTIES FOLDER "Clang executables")
  set_clang_windows_version_resource_properties(${name})
endmacro(add_clang_executable)

macro(add_clang_tool name)
  if (NOT CLANG_BUILD_TOOLS)
    set(EXCLUDE_FROM_ALL ON)
  endif()

  add_clang_executable(${name} ${ARGN})
  add_dependencies(${name} clang-resource-headers)

  if (CLANG_BUILD_TOOLS)
    get_target_export_arg(${name} Clang export_to_clangtargets)
    install(TARGETS ${name}
      ${export_to_clangtargets}
      RUNTIME DESTINATION bin
      COMPONENT ${name})

    if(NOT LLVM_ENABLE_IDE)
      add_llvm_install_targets(install-${name}
                               DEPENDS ${name}
                               COMPONENT ${name})
    endif()
    set_property(GLOBAL APPEND PROPERTY CLANG_EXPORTS ${name})
  endif()
endmacro()

macro(add_clang_symlink name dest)
  add_llvm_tool_symlink(${name} ${dest} ALWAYS_GENERATE)
  # Always generate install targets
  llvm_install_symlink(${name} ${dest} ALWAYS_GENERATE)
endmacro()

function(clang_target_link_libraries target type)
  if (CLANG_LINK_CLANG_DYLIB)
    target_link_libraries(${target} ${type} clang-cpp)
  else()
    target_link_libraries(${target} ${type} ${ARGN})
  endif()

endfunction()
