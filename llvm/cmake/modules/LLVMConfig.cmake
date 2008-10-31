macro(llvm_config executable link_components)
  if( MSVC )
    msvc_llvm_config(${executable} ${link_components})
  else( MSVC )
    nix_llvm_config(${executable} ${link_components})
  endif( MSVC )
endmacro(llvm_config executable link_components)


function(msvc_llvm_config executable link_components)
  foreach(c ${link_components})
    if( c STREQUAL "jit" )
      set_target_properties(${executable}
	PROPERTIES
	LINK_FLAGS "/INCLUDE:_X86TargetMachineModule")
    endif( c STREQUAL "jit" )
  endforeach(c)
  msvc_map_components_to_libraries(${link_components} LIBRARIES)
  target_link_libraries(${executable} ${LIBRARIES})
endfunction(msvc_llvm_config executable link_components)


function(msvc_map_components_to_libraries link_components out_libs)
  foreach(c ${link_components})
    if( c STREQUAL "native" )
      # TODO: we assume ARCH is X86. In this case, we must use nativecodegen
      # component instead. Do nothing, as in llvm-config script.
    elseif( c STREQUAL "nativecodegen" )
      # TODO: we assume ARCH is X86.
      list(APPEND expanded_components "LLVMX86CodeGen")
    elseif( c STREQUAL "backend" )
      # same case as in `native'.
    elseif( c STREQUAL "engine" )
      # TODO: as we assume we are on X86, this is `jit'.
      list(APPEND expanded_components "LLVMJIT")
    elseif( c STREQUAL "X86" )
      # TODO: we assume we are on X86.
      list(APPEND expanded_components "LLVMX86CodeGen")
      list(APPEND expanded_components "LLVMX86AsmPrinter")
    elseif( c STREQUAL "all" )
      list(APPEND expanded_components ${llvm_libs})
    else( c STREQUAL "native" )
      list(APPEND expanded_components LLVM${c})
    endif( c STREQUAL "native" )
  endforeach(c)
  # We must match capitalization.
  string(TOUPPER "${llvm_libs}" capitalized_libs)
  set(curr_idx 0)
  list(LENGTH expanded_components lst_size)
  while( ${curr_idx} LESS ${lst_size} )
    list(GET expanded_components ${curr_idx} c)
    string(TOUPPER "${c}" capitalized)
    list(FIND capitalized_libs ${capitalized} idx)
    if( idx LESS 0 )
      message(FATAL_ERROR "Library ${c} not found in list of llvm libraries.")
    endif( idx LESS 0 )
    list(GET llvm_libs ${idx} canonical_lib)
    list(APPEND result ${canonical_lib})
    list(APPEND result ${MSVC_LIB_DEPS_${canonical_lib}})
    list(APPEND expanded_components ${MSVC_LIB_DEPS_${canonical_lib}})
    list(LENGTH expanded_components lst_size)
    math(EXPR curr_idx "${curr_idx} + 1")
  endwhile( ${curr_idx} LESS ${lst_size} )
  list(REMOVE_DUPLICATES result)
  set(${out_libs} ${result} PARENT_SCOPE)
endfunction(msvc_map_components_to_libraries)


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


# This data is used on MSVC for stablishing executable/library
# dependencies.  Comes from the llvm-config script, which is built and
# installed on the bin directory for MinGW or Linux. At the end of the
# script, you'll see lines like this:
# 
# LLVMARMAsmPrinter.o: LLVMARMCodeGen.o libLLVMAsmPrinter.a
# libLLVMCodeGen.a libLLVMCore.a libLLVMSupport.a libLLVMTarget.a
# 
# This is translated to:
# 
# set(MSVC_LIB_DEPS_LLVMARMAsmPrinter LLVMARMCodeGen LLVMAsmPrinter
# LLVMCodeGen LLVMCore LLVMSupport LLVMTarget)
# 
# It is necessary to remove the `lib' prefix, the `.a' and `.o'
# suffixes.  Watch out for this line:
# 
# LLVMExecutionEngine.o LLVMJIT.o: libLLVMCodeGen.a libLLVMCore.a
# libLLVMSupport.a libLLVMSystem.a libLLVMTarget.a
# 
# See how there are two elements before the colon. This must be
# translated as if it were:
# 
# LLVMExecutionEngine.o: libLLVMCodeGen.a libLLVMCore.a
# libLLVMSupport.a libLLVMSystem.a libLLVMTarget.a LLVMJIT.o:
# libLLVMCodeGen.a libLLVMCore.a libLLVMSupport.a libLLVMSystem.a
# libLLVMTarget.a
# 
# TODO: do this transformations on cmake.
# 
# It is very important that the LLVM built for extracting this data
# must contain all targets, not just X86.


set(MSVC_LIB_DEPS_LLVMARMAsmPrinter LLVMARMCodeGen LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSupport LLVMTarget)
set(MSVC_LIB_DEPS_LLVMARMCodeGen LLVMCodeGen LLVMCore LLVMSelectionDAG LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMAlpha LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSelectionDAG LLVMSupport LLVMTarget)
set(MSVC_LIB_DEPS_LLVMCBackend LLVMAnalysis LLVMCodeGen LLVMCore LLVMScalarOpts LLVMSupport LLVMTarget LLVMTransformUtils LLVMipa)
set(MSVC_LIB_DEPS_LLVMCBase LLVMSupport)
set(MSVC_LIB_DEPS_LLVMCellSPU LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSelectionDAG LLVMSupport LLVMTarget)
set(MSVC_LIB_DEPS_LLVMCppBackend LLVMCore LLVMSupport LLVMTarget)
set(MSVC_LIB_DEPS_LLVMExecutionEngine LLVMCodeGen LLVMCore LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMJIT LLVMCodeGen LLVMCore LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMIA64 LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSelectionDAG LLVMSupport LLVMTarget)
set(MSVC_LIB_DEPS_LLVMInterpreter LLVMExecutionEngine LLVMCodeGen LLVMCore LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMMSIL LLVMAnalysis LLVMCodeGen LLVMCore LLVMScalarOpts LLVMSupport LLVMTarget LLVMTransformUtils LLVMipa)
set(MSVC_LIB_DEPS_LLVMMips LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSelectionDAG LLVMSupport LLVMTarget)
set(MSVC_LIB_DEPS_LLVMPIC16 LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSelectionDAG LLVMSupport LLVMTarget)
set(MSVC_LIB_DEPS_LLVMPowerPCAsmPrinter LLVMPowerPCCodeGen LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSupport LLVMTarget)
set(MSVC_LIB_DEPS_LLVMPowerPCCodeGen LLVMCodeGen LLVMCore LLVMSelectionDAG LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMSparc LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSelectionDAG LLVMSupport LLVMTarget)
set(MSVC_LIB_DEPS_LLVMX86AsmPrinter LLVMX86CodeGen LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSupport LLVMTarget)
set(MSVC_LIB_DEPS_LLVMX86CodeGen LLVMCodeGen LLVMCore LLVMSelectionDAG LLVMSupport LLVMTarget)
set(MSVC_LIB_DEPS_LLVMAnalysis LLVMCore LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMArchive LLVMBitReader LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMAsmParser LLVMCore LLVMSupport)
set(MSVC_LIB_DEPS_LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMBitReader LLVMCore LLVMSupport)
set(MSVC_LIB_DEPS_LLVMBitWriter LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMCodeGen LLVMAnalysis LLVMCore LLVMScalarOpts LLVMSupport LLVMSystem LLVMTarget LLVMTransformUtils)
set(MSVC_LIB_DEPS_LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMDebugger LLVMAnalysis LLVMBitReader LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMInstrumentation LLVMCore LLVMScalarOpts LLVMSupport LLVMTransformUtils)
set(MSVC_LIB_DEPS_LLVMLinker LLVMArchive LLVMBitReader LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMScalarOpts LLVMAnalysis LLVMCore LLVMSupport LLVMTarget LLVMTransformUtils)
set(MSVC_LIB_DEPS_LLVMSelectionDAG LLVMAnalysis LLVMCodeGen LLVMCore LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMSystem )
set(MSVC_LIB_DEPS_LLVMTarget LLVMCore LLVMSupport)
set(MSVC_LIB_DEPS_LLVMTransformUtils LLVMAnalysis LLVMCore LLVMSupport LLVMTarget LLVMipa)
set(MSVC_LIB_DEPS_LLVMipa LLVMAnalysis LLVMCore LLVMSupport)
set(MSVC_LIB_DEPS_LLVMipo LLVMAnalysis LLVMCore LLVMSupport LLVMTarget LLVMTransformUtils LLVMipa)
