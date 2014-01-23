set(LLVM_LINK_COMPONENTS support)
add_definitions(-DLLI_BUILDING_CHILD)

add_llvm_executable(lli-child-target
  ChildTarget.cpp
  ../RemoteTarget.cpp
)
