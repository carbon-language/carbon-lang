set(LLVM_LINK_COMPONENTS
  Object
  Support
  )

add_llvm_utility(obj2yaml
  obj2yaml.cpp coff2yaml.cpp
  )
