include(LLVMProcessSources)
include(LLVMConfig)

macro(add_llvm_library name)
  llvm_process_sources( ALL_FILES ${ARGN} )
  add_library( ${name} ${ALL_FILES} )
  set_property( GLOBAL APPEND PROPERTY LLVM_LIBS ${name} )
  set_property( GLOBAL APPEND PROPERTY LLVM_LIB_TARGETS ${name} )
  if( LLVM_COMMON_DEPENDS )
    add_dependencies( ${name} ${LLVM_COMMON_DEPENDS} )
  endif( LLVM_COMMON_DEPENDS )

  if( BUILD_SHARED_LIBS )
    get_system_libs(sl)
    target_link_libraries( ${name} ${sl} )
  endif()

  install(TARGETS ${name}
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX})
  # The LLVM Target library shall be built before its sublibraries
  # (asmprinter, etc) because those may use tablegenned files which
  # generation is triggered by the main LLVM target library. Necessary
  # for parallel builds:
  if( CURRENT_LLVM_TARGET )
    add_dependencies(${name} ${CURRENT_LLVM_TARGET})
  endif()
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

    if (APPLE)
      # Darwin-specific linker flags for loadable modules.
      set_target_properties(${name} PROPERTIES
        LINK_FLAGS "-Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
    endif()

    install(TARGETS ${name}
      LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX})
  endif()
endmacro(add_llvm_loadable_module name)


macro(add_llvm_executable name)
  llvm_process_sources( ALL_FILES ${ARGN} )
  if( EXCLUDE_FROM_ALL )
    add_executable(${name} EXCLUDE_FROM_ALL ${ALL_FILES})
  else()
    add_executable(${name} ${ALL_FILES})
  endif()
  set(EXCLUDE_FROM_ALL OFF)
  if( LLVM_USED_LIBS )
    foreach(lib ${LLVM_USED_LIBS})
      target_link_libraries( ${name} ${lib} )
    endforeach(lib)
  endif( LLVM_USED_LIBS )
  if( LLVM_LINK_COMPONENTS )
    llvm_config(${name} ${LLVM_LINK_COMPONENTS})
  endif( LLVM_LINK_COMPONENTS )
  if( LLVM_COMMON_DEPENDS )
    add_dependencies( ${name} ${LLVM_COMMON_DEPENDS} )
  endif( LLVM_COMMON_DEPENDS )
  if( NOT MINGW )
    get_system_libs(llvm_system_libs)
    if( llvm_system_libs )
      target_link_libraries(${name} ${llvm_system_libs})
    endif()
  endif()
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
endmacro(add_llvm_example name)


macro(add_llvm_target target_name)
  if( TABLEGEN_OUTPUT )
    add_custom_target(${target_name}Table_gen
      DEPENDS ${TABLEGEN_OUTPUT})
    add_dependencies(${target_name}Table_gen ${LLVM_COMMON_DEPENDS})
  endif( TABLEGEN_OUTPUT )
  include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR})
  add_llvm_library(LLVM${target_name} ${ARGN} ${TABLEGEN_OUTPUT})
  if ( TABLEGEN_OUTPUT )
    add_dependencies(LLVM${target_name} ${target_name}Table_gen)
  endif (TABLEGEN_OUTPUT)
  set( CURRENT_LLVM_TARGET LLVM${target_name} )
endmacro(add_llvm_target)
