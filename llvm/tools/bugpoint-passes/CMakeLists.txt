if( NOT LLVM_BUILD_TOOLS )
  set(EXCLUDE_FROM_ALL ON)
endif()

add_llvm_loadable_module( BugpointPasses
  TestPasses.cpp
  )

add_dependencies(BugpointPasses bugpoint)
