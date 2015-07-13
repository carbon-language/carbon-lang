add_llvm_tool_subdirectory(llvm-config)

# Build polly before the tools: the tools link against polly when
# LINK_POLLY_INTO_TOOLS is set.
if(WITH_POLLY)
  add_llvm_external_project(polly)
else(WITH_POLLY)
  list(APPEND LLVM_IMPLICIT_PROJECT_IGNORE "${LLVM_MAIN_SRC_DIR}/tools/polly")
endif(WITH_POLLY)

if( LLVM_BUILD_LLVM_DYLIB )
  add_llvm_tool_subdirectory(llvm-shlib)
else()
  ignore_llvm_tool_subdirectory(llvm-shlib)
endif()

add_llvm_tool_subdirectory(opt)
add_llvm_tool_subdirectory(llvm-as)
add_llvm_tool_subdirectory(llvm-dis)
add_llvm_tool_subdirectory(llvm-mc)

add_llvm_tool_subdirectory(llc)
add_llvm_tool_subdirectory(llvm-ar)
add_llvm_tool_subdirectory(llvm-nm)
add_llvm_tool_subdirectory(llvm-size)

add_llvm_tool_subdirectory(llvm-cov)
add_llvm_tool_subdirectory(llvm-profdata)
add_llvm_tool_subdirectory(llvm-link)
add_llvm_tool_subdirectory(lli)

add_llvm_tool_subdirectory(llvm-extract)
add_llvm_tool_subdirectory(llvm-diff)
add_llvm_tool_subdirectory(macho-dump)
add_llvm_tool_subdirectory(llvm-objdump)
add_llvm_tool_subdirectory(llvm-readobj)
add_llvm_tool_subdirectory(llvm-rtdyld)
add_llvm_tool_subdirectory(llvm-dwarfdump)
add_llvm_tool_subdirectory(dsymutil)
add_llvm_tool_subdirectory(llvm-cxxdump)
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

add_llvm_tool_subdirectory(verify-uselistorder)

add_llvm_tool_subdirectory(llvm-symbolizer)

add_llvm_tool_subdirectory(llvm-c-test)

add_llvm_tool_subdirectory(obj2yaml)
add_llvm_tool_subdirectory(yaml2obj)

add_llvm_tool_subdirectory(llvm-go)

add_llvm_tool_subdirectory(llvm-pdbdump)

if(NOT CYGWIN AND LLVM_ENABLE_PIC)
  add_llvm_tool_subdirectory(lto)
  add_llvm_tool_subdirectory(llvm-lto)
else()
  ignore_llvm_tool_subdirectory(lto)
  ignore_llvm_tool_subdirectory(llvm-lto)
endif()

add_llvm_tool_subdirectory(gold)

add_llvm_external_project(clang)
add_llvm_external_project(llgo)
add_llvm_external_project(lld)
add_llvm_external_project(lldb)

# Automatically add remaining sub-directories containing a 'CMakeLists.txt'
# file as external projects.
add_llvm_implicit_external_projects()

set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} PARENT_SCOPE)
