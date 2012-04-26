include(LLVMProcessSources)
include(LLVM-Config)

macro(add_llvm_library name)
  llvm_process_sources( ALL_FILES ${ARGN} )
  add_library( ${name} ${ALL_FILES} )
  set_property( GLOBAL APPEND PROPERTY LLVM_LIBS ${name} )
  if( LLVM_COMMON_DEPENDS )
    add_dependencies( ${name} ${LLVM_COMMON_DEPENDS} )
  endif( LLVM_COMMON_DEPENDS )

  if( BUILD_SHARED_LIBS )
    llvm_config( ${name} ${LLVM_LINK_COMPONENTS} )
  endif()

  # Ensure that the system libraries always comes last on the
  # list. Without this, linking the unit tests on MinGW fails.
  link_system_libs( ${name} )

  if( EXCLUDE_FROM_ALL )
    set_target_properties( ${name} PROPERTIES EXCLUDE_FROM_ALL ON)
  else()
    install(TARGETS ${name}
      LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX})
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
    set_target_properties( ${name} PROPERTIES PREFIX "" )

    llvm_config( ${name} ${LLVM_LINK_COMPONENTS} )
    link_system_libs( ${name} )

    if (APPLE)
      # Darwin-specific linker flags for loadable modules.
      set_target_properties(${name} PROPERTIES
        LINK_FLAGS "-Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
    endif()

    if( EXCLUDE_FROM_ALL )
      set_target_properties( ${name} PROPERTIES EXCLUDE_FROM_ALL ON)
    else()
      install(TARGETS ${name}
	LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
	ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX})
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
  set(EXCLUDE_FROM_ALL OFF)
  target_link_libraries( ${name} ${LLVM_USED_LIBS} )
  llvm_config( ${name} ${LLVM_LINK_COMPONENTS} )
  if( LLVM_COMMON_DEPENDS )
    add_dependencies( ${name} ${LLVM_COMMON_DEPENDS} )
  endif( LLVM_COMMON_DEPENDS )
  link_system_libs( ${name} )
endmacro(add_llvm_executable name)


macro(add_llvm_tool name)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LLVM_TOOLS_BINARY_DIR})
  if( NOT LLVM_BUILD_TOOLS )
    set(EXCLUDE_FROM_ALL ON)
  endif()
  add_llvm_executable(${name} ${ARGN})
  if( LLVM_BUILD_TOOLS )
    install(TARGETS ${name} RUNTIME DESTINATION bin)
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
# enable or disable building it with everthing else.
macro(add_llvm_external_project name)
  string(TOUPPER ${name} nameUPPER)
  set(LLVM_EXTERNAL_${nameUPPER}_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${name}"
      CACHE PATH "Path to ${name} source directory")
  if (NOT ${LLVM_EXTERNAL_${nameUPPER}_SOURCE_DIR} STREQUAL ""
      AND EXISTS ${LLVM_EXTERNAL_${nameUPPER}_SOURCE_DIR}/CMakeLists.txt)
    option(LLVM_EXTERNAL_${nameUPPER}_BUILD
           "Whether to build ${name} as part of LLVM" ON)
    if (LLVM_EXTERNAL_${nameUPPER}_BUILD)
      add_subdirectory(${LLVM_EXTERNAL_${nameUPPER}_SOURCE_DIR} ${name})
    endif()
  endif()
endmacro(add_llvm_external_project)
