include(AddFileDependencies)

function(llvm_process_sources)
  set( sources ${ARGN} )
  # Create file dependencies on the tablegenned files, if any.  Seems
  # that this is not strictly needed, as dependencies of the .cpp
  # sources on the tablegenned .inc files are detected and handled,
  # but just in case...
  foreach( s ${sources} )
    set( f ${CMAKE_CURRENT_SOURCE_DIR}/${s} )
    add_file_dependencies( ${f} ${TABLEGEN_OUTPUT} )
  endforeach(s)
endfunction(llvm_process_sources)
