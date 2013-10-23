# NOTE: The tools are organized into groups of four consisting of one large and
# three small executables. This is done to minimize memory load in parallel
# builds.  Please retain this ordering.

add_llvm_tool_subdirectory(llvm-config)

add_llvm_tool_subdirectory(opt)
add_llvm_tool_subdirectory(llvm-as)
add_llvm_tool_subdirectory(llvm-dis)
add_llvm_tool_subdirectory(llvm-mc)

add_llvm_tool_subdirectory(llc)
add_llvm_tool_subdirectory(llvm-ar)
add_llvm_tool_subdirectory(llvm-nm)
add_llvm_tool_subdirectory(llvm-size)

add_llvm_tool_subdirectory(llvm-cov)
add_llvm_tool_subdirectory(llvm-link)
add_llvm_tool_subdirectory(lli)

add_llvm_tool_subdirectory(llvm-extract)
add_llvm_tool_subdirectory(llvm-diff)
add_llvm_tool_subdirectory(macho-dump)
add_llvm_tool_subdirectory(llvm-objdump)
add_llvm_tool_subdirectory(llvm-readobj)
add_llvm_tool_subdirectory(llvm-rtdyld)
add_llvm_tool_subdirectory(llvm-dwarfdump)
if( LLVM_USE_INTEL_JITEVENTS )
  add_llvm_tool_subdirectory(llvm-jitlistener)
else()
  ignore_llvm_tool_subdirectory(llvm-jitlistener)
endif( LLVM_USE_INTEL_JITEVENTS )

add_llvm_tool_subdirectory(bugpoint)
add_llvm_tool_subdirectory(bugpoint-passes)
add_llvm_tool_subdirectory(llvm-bcanalyzer)
add_llvm_tool_subdirectory(llvm-stress)
add_llvm_tool_subdirectory(llvm-mcmarkup)

add_llvm_tool_subdirectory(llvm-symbolizer)

add_llvm_tool_subdirectory(llvm-c-test)

add_llvm_tool_subdirectory(obj2yaml)
add_llvm_tool_subdirectory(yaml2obj)

if( NOT CYGWIN )
  add_llvm_tool_subdirectory(lto)
  add_llvm_tool_subdirectory(llvm-lto)
else()
  ignore_llvm_tool_subdirectory(lto)
  ignore_llvm_tool_subdirectory(llvm-lto)
endif()

if( LLVM_ENABLE_PIC )
  # TODO: support other systems:
  if( (CMAKE_SYSTEM_NAME STREQUAL "Linux")
      OR (CMAKE_SYSTEM_NAME STREQUAL "FreeBSD") )
    add_llvm_tool_subdirectory(gold)
  else()
    ignore_llvm_tool_subdirectory(gold)
  endif()
else()
  ignore_llvm_tool_subdirectory(gold)
endif()

add_llvm_external_project(clang)

if( NOT LLVM_INCLUDE_TOOLS STREQUAL "bootstrap-only" )
  add_llvm_external_project(lld)
  add_llvm_external_project(lldb)
  add_llvm_external_project(polly)

  # Automatically add remaining sub-directories containing a 'CMakeLists.txt'
  # file as external projects.
  add_llvm_implicit_external_projects()
endif()

set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} PARENT_SCOPE)
