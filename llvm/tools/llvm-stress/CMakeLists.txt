set(LLVM_LINK_COMPONENTS
  Core
  IPA
  Support
  )

add_llvm_tool(llvm-stress
  llvm-stress.cpp
  )
set_target_properties(llvm-stress PROPERTIES ENABLE_EXPORTS 1)
