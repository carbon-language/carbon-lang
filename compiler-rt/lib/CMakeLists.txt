# First, add the subdirectories which contain feature-based runtime libraries
# and several convenience helper libraries.

include(AddCompilerRT)
include(SanitizerUtils)

if(COMPILER_RT_BUILD_BUILTINS)
  add_subdirectory(builtins)
endif()

if(COMPILER_RT_BUILD_SANITIZERS)
  message(STATUS "Adding runtimes...")
  foreach(runtime ${COMPILER_RT_RUNTIMES_TO_BUILD})
    message(STATUS "Adding runtime: ${runtime}")
    add_subdirectory(${runtime})
  endforeach()
endif()
