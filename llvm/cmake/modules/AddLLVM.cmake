include(LLVMConfig)

macro(add_llvm_library name)
  add_library( ${name} ${ARGN} )
  set( llvm_libs ${llvm_libs} ${name} PARENT_SCOPE)
endmacro(add_llvm_library name)


macro(add_llvm_executable name)
  add_executable(${name} ${ARGN})
  if( LLVM_LINK_COMPONENTS )
    llvm_config(${name} ${LLVM_LINK_COMPONENTS})
  endif( LLVM_LINK_COMPONENTS )
  if( MSVC )
    target_link_libraries(${name} ${llvm_libs})
  else( MSVC )
    add_dependencies(${name} llvm-config.target)
    set_target_properties(${name}
      PROPERTIES
      LINK_FLAGS "-L ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
    if( MINGW )
      target_link_libraries(${name} DbgHelp psapi)
    elseif( CMAKE_HOST_UNIX )
      target_link_libraries(${name} dl)
    endif( MINGW )
  endif( MSVC )
endmacro(add_llvm_executable name)


macro(add_llvm_tool name)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LLVM_TOOLS_BINARY_DIR})
  add_llvm_executable(${name} ${ARGN})
endmacro(add_llvm_tool name)


macro(add_llvm_example name)
#  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LLVM_EXAMPLES_BINARY_DIR})
  add_llvm_executable(${name} ${ARGN})
endmacro(add_llvm_example name)
