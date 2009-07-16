include(FindPerl)


function(get_system_libs return_var)
  # Returns in `return_var' a list of system libraries used by LLVM.
  if( NOT MSVC )
    if( MINGW )
      set(system_libs ${system_libs} imagehlp psapi)
    elseif( CMAKE_HOST_UNIX )
      if( HAVE_LIBDL )
	set(system_libs ${system_libs} dl)
      endif()
      if( LLVM_ENABLE_THREADS AND HAVE_LIBPTHREAD )
	set(system_libs ${system_libs} pthread)
      endif()
    endif( MINGW )
  endif( NOT MSVC )
  set(${return_var} ${system_libs} PARENT_SCOPE)
endfunction(get_system_libs)


macro(llvm_config executable)
  explicit_llvm_config(${executable} ${ARGN})
endmacro(llvm_config)


function(explicit_llvm_config executable)
  set( link_components ${ARGN} )

  explicit_map_components_to_libraries(LIBRARIES ${link_components})
  target_link_libraries(${executable} ${LIBRARIES})
endfunction(explicit_llvm_config)


function(explicit_map_components_to_libraries out_libs)
  set( link_components ${ARGN} )
  foreach(c ${link_components})
    # add codegen/asmprinter
    list(FIND LLVM_TARGETS_TO_BUILD ${c} idx)
    if( NOT idx LESS 0 )
      list(FIND llvm_libs "LLVM${c}CodeGen" idx)
      if( NOT idx LESS 0 )
	list(APPEND expanded_components "LLVM${c}CodeGen")
      else()
	list(FIND llvm_libs "LLVM${c}" idx)
	if( NOT idx LESS 0 )
	  list(APPEND expanded_components "LLVM${c}")
	else()
	  message(FATAL_ERROR "Target ${c} is not in the set of libraries.")
	endif()
      endif()
      list(FIND llvm_libs "LLVM${c}AsmPrinter" asmidx)
      if( NOT asmidx LESS 0 )
        list(APPEND expanded_components "LLVM${c}AsmPrinter")
      endif()
      list(FIND llvm_libs "LLVM${c}Info" asmidx)
      if( NOT asmidx LESS 0 )
        list(APPEND expanded_components "LLVM${c}Info")
      endif()
    elseif( c STREQUAL "native" )
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
    elseif( c STREQUAL "all" )
      list(APPEND expanded_components ${llvm_libs})
    else( NOT idx LESS 0 )
      list(APPEND expanded_components LLVM${c})
    endif( NOT idx LESS 0 )
  endforeach(c)
  # We must match capitalization.
  string(TOUPPER "${llvm_libs}" capitalized_libs)
  list(REMOVE_DUPLICATES expanded_components)
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
    list(REMOVE_DUPLICATES expanded_components)
    list(LENGTH expanded_components lst_size)
    math(EXPR curr_idx "${curr_idx} + 1")
  endwhile( ${curr_idx} LESS ${lst_size} )
  list(REMOVE_DUPLICATES result)
  set(${out_libs} ${result} PARENT_SCOPE)
endfunction(explicit_map_components_to_libraries)

# This data is used to establish executable/library
# dependencies.  Comes from the llvm-config script, which is built and
# installed on the bin directory for MinGW or Linux. At the end of the
# script, you'll see lines like this:

# LLVMARMAsmPrinter.o: LLVMARMCodeGen.o libLLVMAsmPrinter.a libLLVMCodeGen.a libLLVMCore.a libLLVMSupport.a libLLVMTarget.a

# This is translated to:

# set(MSVC_LIB_DEPS_LLVMARMAsmPrinter LLVMARMCodeGen LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSupport LLVMTarget)

# It is necessary to remove the `lib' prefix and the `.a'.

# This 'sed' script should do the trick:
# sed -e s'#\.a##g' -e 's#libLLVM#LLVM#g' -e 's#: ##' -e 's#\(.*\)#set(MSVC_LIB_DEPS_\1)#' ~/llvm/tools/llvm-config/LibDeps.txt
#

# TODO: do this transformations on cmake.

# It is very important that the LLVM built for extracting this data
# must contain all targets, not just X86.


set(MSVC_LIB_DEPS_LLVMARMAsmPrinter LLVMARMInfo LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMARMCodeGen LLVMARMInfo LLVMCodeGen LLVMCore LLVMSelectionDAG LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMARMInfo LLVMCore LLVMSupport)
set(MSVC_LIB_DEPS_LLVMAlphaAsmPrinter LLVMAlphaInfo LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMAlphaCodeGen LLVMAlphaInfo LLVMCodeGen LLVMCore LLVMSelectionDAG LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMAlphaInfo LLVMCore LLVMSupport)
set(MSVC_LIB_DEPS_LLVMAnalysis LLVMCore LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMArchive LLVMBitReader LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMAsmParser LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMAsmPrinter LLVMAnalysis LLVMCodeGen LLVMCore LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMBitReader LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMBitWriter LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMCBackend LLVMAnalysis LLVMCBackendInfo LLVMCodeGen LLVMCore LLVMScalarOpts LLVMSupport LLVMSystem LLVMTarget LLVMTransformUtils LLVMipa)
set(MSVC_LIB_DEPS_LLVMCBackendInfo LLVMSupport)
set(MSVC_LIB_DEPS_LLVMCellSPUAsmPrinter LLVMAsmPrinter LLVMCellSPUInfo LLVMCodeGen LLVMCore LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMCellSPUCodeGen LLVMCellSPUInfo LLVMCodeGen LLVMCore LLVMSelectionDAG LLVMSupport LLVMTarget)
set(MSVC_LIB_DEPS_LLVMCellSPUInfo LLVMSupport)
set(MSVC_LIB_DEPS_LLVMCodeGen LLVMAnalysis LLVMCore LLVMScalarOpts LLVMSupport LLVMSystem LLVMTarget LLVMTransformUtils)
set(MSVC_LIB_DEPS_LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMCppBackend LLVMCore LLVMCppBackendInfo LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMCppBackendInfo LLVMSupport)
set(MSVC_LIB_DEPS_LLVMDebugger LLVMAnalysis LLVMBitReader LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMExecutionEngine LLVMCore LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMIA64AsmPrinter LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMIA64Info LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMIA64CodeGen LLVMCodeGen LLVMCore LLVMIA64Info LLVMSelectionDAG LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMIA64Info LLVMSupport)
set(MSVC_LIB_DEPS_LLVMInstrumentation LLVMCore LLVMScalarOpts LLVMSupport LLVMSystem LLVMTransformUtils)
set(MSVC_LIB_DEPS_LLVMInterpreter LLVMCodeGen LLVMCore LLVMExecutionEngine LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMJIT LLVMCodeGen LLVMCore LLVMExecutionEngine LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMLinker LLVMArchive LLVMBitReader LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMMC LLVMSupport)
set(MSVC_LIB_DEPS_LLVMMSIL LLVMAnalysis LLVMCodeGen LLVMCore LLVMMSILInfo LLVMScalarOpts LLVMSupport LLVMSystem LLVMTarget LLVMTransformUtils LLVMipa)
set(MSVC_LIB_DEPS_LLVMMSILInfo LLVMSupport)
set(MSVC_LIB_DEPS_LLVMMSP430 LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMMSP430Info LLVMSelectionDAG LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMMSP430Info LLVMSupport)
set(MSVC_LIB_DEPS_LLVMMipsAsmPrinter LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMMipsCodeGen LLVMMipsInfo LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMMipsCodeGen LLVMCodeGen LLVMCore LLVMMipsInfo LLVMSelectionDAG LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMMipsInfo LLVMSupport)
set(MSVC_LIB_DEPS_LLVMPIC16 LLVMAnalysis LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMPIC16Info LLVMSelectionDAG LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMPIC16Info LLVMSupport)
set(MSVC_LIB_DEPS_LLVMPowerPCAsmPrinter LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMPowerPCInfo LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMPowerPCCodeGen LLVMCodeGen LLVMCore LLVMPowerPCInfo LLVMSelectionDAG LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMPowerPCInfo LLVMCore LLVMSupport)
set(MSVC_LIB_DEPS_LLVMScalarOpts LLVMAnalysis LLVMCore LLVMSupport LLVMSystem LLVMTarget LLVMTransformUtils)
set(MSVC_LIB_DEPS_LLVMSelectionDAG LLVMAnalysis LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMSparcAsmPrinter LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSparcInfo LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMSparcCodeGen LLVMCodeGen LLVMCore LLVMSelectionDAG LLVMSparcInfo LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMSparcInfo LLVMCore LLVMSupport)
set(MSVC_LIB_DEPS_LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMSystem)
set(MSVC_LIB_DEPS_LLVMTarget LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMTransformUtils LLVMAnalysis LLVMCore LLVMSupport LLVMSystem LLVMTarget LLVMipa)
set(MSVC_LIB_DEPS_LLVMX86AsmPrinter LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMMC LLVMSupport LLVMSystem LLVMTarget LLVMX86CodeGen LLVMX86Info)
set(MSVC_LIB_DEPS_LLVMX86CodeGen LLVMCodeGen LLVMCore LLVMSelectionDAG LLVMSupport LLVMSystem LLVMTarget LLVMX86Info)
set(MSVC_LIB_DEPS_LLVMX86Info LLVMCore LLVMSupport)
set(MSVC_LIB_DEPS_LLVMXCore LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSelectionDAG LLVMSupport LLVMSystem LLVMTarget LLVMXCoreInfo)
set(MSVC_LIB_DEPS_LLVMXCoreInfo LLVMSupport)
set(MSVC_LIB_DEPS_LLVMipa LLVMAnalysis LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMipo LLVMAnalysis LLVMCore LLVMSupport LLVMSystem LLVMTarget LLVMTransformUtils LLVMipa)
