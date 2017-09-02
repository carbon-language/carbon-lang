add_subdirectory(merge-fdata)
add_subdirectory(Passes)

# Get the current git revision for BOLT.
function(get_version ofn)
  find_program(git_executable NAMES git git.exe git.cmd)
  if (git_executable)
    execute_process(COMMAND ${git_executable} rev-parse HEAD
      WORKING_DIRECTORY ${LLVM_MAIN_SRC_DIR}
      TIMEOUT 5
      RESULT_VARIABLE git_result
      OUTPUT_VARIABLE git_output)
      if( git_result EQUAL 0 )
        string(STRIP "${git_output}" git_ref_id)
        set(BOLT_REVISION "${git_ref_id}")
      endif()
  endif()

  # If we can't find a revision, set it to "<unknown>".
  if (NOT BOLT_REVISION)
    set(BOLT_REVISION "<unknown>")
  endif()

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
    COMMAND echo '"${BOLT_REVISION}"' > ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
    COMMENT "Generating bogus ${ofn}..."
  )

  set(VERSION_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ofn} PARENT_SCOPE)

  # `make clean' must remove all those generated files:
  set_property(DIRECTORY APPEND
    PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${ofn})
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${ofn} PROPERTIES
    GENERATED 1)
endfunction()

# Creates a public target for generating the revision file.
function(add_public_gen_version_target target)
  add_custom_target(${target} DEPENDS ${VERSION_OUTPUT})
  set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} ${target} PARENT_SCOPE)
endfunction()

get_version(BoltRevision.inc)
add_public_gen_version_target(GenBoltRevision)

set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  BOLTPasses
  CodeGen
  Core
  DebugInfoDWARF
  MC
  MCDisassembler
  MCParser
  Object
  Orcjit
  Support
  )

add_llvm_tool(llvm-bolt
  llvm-bolt.cpp
  BinaryBasicBlock.cpp
  BinaryContext.cpp
  BinaryFunction.cpp
  BinaryPassManager.cpp
  CalcCacheMetrics.cpp
  DataAggregator.cpp
  DataReader.cpp
  DebugData.cpp
  DWARFRewriter.cpp
  Exceptions.cpp
  RewriteInstance.cpp
  )

add_llvm_tool_symlink(perf2bolt llvm-bolt)
