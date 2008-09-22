macro(llvm_config executable link_components)
  if( MSVC )
    msvc_llvm_config(${executable} ${link_components})
  else( MSVC )
    nix_llvm_config(${executable} ${link_components})
  endif( MSVC )
endmacro(llvm_config executable link_components)


macro(msvc_llvm_config executable link_components)
  foreach(c ${link_components})
    message(STATUS ${c})
    if( c STREQUAL "jit" )
      message(STATUS "linking jit")
      set_target_properties(${executable}
	PROPERTIES
	LINK_FLAGS "/INCLUDE:_X86TargetMachineModule")
    endif( c STREQUAL "jit" )
  endforeach(c)
  target_link_libraries(${executable} ${llvm_libs})
endmacro(msvc_llvm_config executable link_components)


macro(nix_llvm_config executable link_components)
  set(lc "")
  foreach(c ${LLVM_LINK_COMPONENTS})
    set(lc "${lc} ${c}")
  endforeach(c)
  if( NOT HAVE_LLVM_CONFIG )
    target_link_libraries(${executable}
      "`${LLVM_TOOLS_BINARY_DIR}/llvm-config --libs ${lc}`")
  else( NOT HAVE_LLVM_CONFIG )
    # tbi: Error handling.
    if( NOT PERL_FOUND )
      message(FATAL_ERROR "Perl required but not found!")
    endif( NOT PERL_FOUND )
    execute_process(
      COMMAND sh -c "${PERL_EXECUTABLE} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/llvm-config --libs ${lc}"
      RESULT_VARIABLE rv
      OUTPUT_VARIABLE libs
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT rv EQUAL 0)
      message(FATAL_ERROR "llvm-config failed for executable ${executable}")
    endif(NOT rv EQUAL 0)
    string(REPLACE " " ";" libs ${libs})
    foreach(c ${libs})
      if(c MATCHES ".*\\.o")
	get_filename_component(fn ${c} NAME)
	target_link_libraries(${executable}
	  ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${fn})
      else(c MATCHES ".*\\.o")
	string(REPLACE "-l" "" fn ${c})
	target_link_libraries(${executable} ${fn})
      endif(c MATCHES ".*\\.o")
    endforeach(c)
  endif( NOT HAVE_LLVM_CONFIG )
endmacro(nix_llvm_config executable link_components)
