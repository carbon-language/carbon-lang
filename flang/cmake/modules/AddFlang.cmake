function(flang_tablegen)
  # Syntax:
  # flang_tablegen output-file [tablegen-arg ...] SOURCE source-file
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
    message(FATAL_ERROR "SOURCE source-file required by flang_tablegen")
  endif()

  set( LLVM_TARGET_DEFINITIONS ${CTG_SOURCE} )
  tablegen(FLANG ${CTG_UNPARSED_ARGUMENTS})

  if(CTG_TARGET)
    add_public_tablegen_target(${CTG_TARGET})
    set_target_properties( ${CTG_TARGET} PROPERTIES FOLDER "FLANG tablegenning")
    set_property(GLOBAL APPEND PROPERTY FLANG_TABLEGEN_TARGETS ${CTG_TARGET})
  endif()
endfunction(flang_tablegen)

macro(set_flang_windows_version_resource_properties name)
  if(DEFINED windows_resource_file)
    set_windows_version_resource_properties(${name} ${windows_resource_file}
      VERSION_MAJOR ${FLANG_VERSION_MAJOR}
      VERSION_MINOR ${FLANG_VERSION_MINOR}
      VERSION_PATCHLEVEL ${FLANG_VERSION_PATCHLEVEL}
      VERSION_STRING "${FLANG_VERSION} (${BACKEND_PACKAGE_STRING})"
      PRODUCT_NAME "flang")
  endif()
endmacro()

macro(add_flang_subdirectory name)
  add_llvm_subdirectory(FLANG TOOL ${name})
endmacro()

macro(add_flang_library name)
  cmake_parse_arguments(ARG
    "SHARED"
    ""
    "ADDITIONAL_HEADERS"
    ${ARGN})
  set(srcs)
  if(MSVC_IDE OR XCODE)
    # Add public headers
    file(RELATIVE_PATH lib_path
      ${FLANG_SOURCE_DIR}/lib/
      ${CMAKE_CURRENT_SOURCE_DIR}
    )
    if(NOT lib_path MATCHES "^[.][.]")
      file( GLOB_RECURSE headers
        ${FLANG_SOURCE_DIR}/include/flang/${lib_path}/*.h
        ${FLANG_SOURCE_DIR}/include/flang/${lib_path}/*.def
      )
      set_source_files_properties(${headers} PROPERTIES HEADER_FILE_ONLY ON)

      file( GLOB_RECURSE tds
        ${FLANG_SOURCE_DIR}/include/flang/${lib_path}/*.td
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
  if(ARG_SHARED)
    set(ARG_ENABLE_SHARED SHARED)
  endif()
  llvm_add_library(${name} ${ARG_ENABLE_SHARED} ${ARG_UNPARSED_ARGUMENTS} ${srcs})

  if(TARGET ${name})
    target_link_libraries(${name} INTERFACE ${LLVM_COMMON_LIBS})

    if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY OR ${name} STREQUAL "libflang")

      if(${name} IN_LIST LLVM_DISTRIBUTION_COMPONENTS OR
          NOT LLVM_DISTRIBUTION_COMPONENTS)
        set(export_to_flangtargets EXPORT FlangTargets)
        set_property(GLOBAL PROPERTY FLANG_HAS_EXPORTS True)
      endif()

      install(TARGETS ${name}
        COMPONENT ${name}
        ${export_to_flangtargets}
        LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
        ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
        RUNTIME DESTINATION bin)

      if (${ARG_SHARED} AND NOT CMAKE_CONFIGURATION_TYPES)
        add_custom_target(install-${name}
                          DEPENDS ${name}
                          COMMAND "${CMAKE_COMMAND}"
                                  -DCMAKE_INSTALL_COMPONENT=${name}
                                  -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
      endif()
    endif()
    set_property(GLOBAL APPEND PROPERTY FLANG_EXPORTS ${name})
  else()
    # Add empty "phony" target
    add_custom_target(${name})
  endif()

  set_target_properties(${name} PROPERTIES FOLDER "FLANG libraries")
  set_flang_windows_version_resource_properties(${name})
endmacro(add_flang_library)

macro(add_flang_executable name)
  add_llvm_executable( ${name} ${ARGN} )
  set_target_properties(${name} PROPERTIES FOLDER "FLANG executables")
  set_flang_windows_version_resource_properties(${name})
endmacro(add_flang_executable)

macro(add_flang_tool name)
  if (NOT FLANG_BUILD_TOOLS)
    set(EXCLUDE_FROM_ALL ON)
  endif()

  add_flang_executable(${name} ${ARGN})

  if (FLANG_BUILD_TOOLS)
    if(${name} IN_LIST LLVM_DISTRIBUTION_COMPONENTS OR
        NOT LLVM_DISTRIBUTION_COMPONENTS)
      set(export_to_flangtargets EXPORT FlangTargets)
      set_property(GLOBAL PROPERTY FLANG_HAS_EXPORTS True)
    endif()

    install(TARGETS ${name}
      ${export_to_flangtargets}
      RUNTIME DESTINATION bin
      COMPONENT ${name})

    if(NOT CMAKE_CONFIGURATION_TYPES)
      add_custom_target(install-${name}
        DEPENDS ${name}
        COMMAND "${CMAKE_COMMAND}"
        -DCMAKE_INSTALL_COMPONENT=${name}
        -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
    endif()
    set_property(GLOBAL APPEND PROPERTY FLANG_EXPORTS ${name})
  endif()
endmacro()

macro(add_flang_symlink name dest)
  add_llvm_tool_symlink(${name} ${dest} ALWAYS_GENERATE)
  # Always generate install targets
  llvm_install_symlink(${name} ${dest} ALWAYS_GENERATE)
endmacro()
