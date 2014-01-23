set(LLVM_LINK_COMPONENTS support)

add_llvm_executable(lli-child-target
  ChildTarget.cpp
  ../RemoteTarget.cpp
)
