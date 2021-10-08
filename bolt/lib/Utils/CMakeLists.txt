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

add_llvm_library(LLVMBOLTUtils
  CommandLineOpts.cpp
  Utils.cpp

  LINK_LIBS
  ${LLVM_PTHREAD_LIB}

  LINK_COMPONENTS
  Support
  )
