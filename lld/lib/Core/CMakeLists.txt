set(LLVM_LINK_COMPONENTS support)

add_lld_library(lldCore
  DefinedAtom.cpp
  Error.cpp
  File.cpp
  InputGraph.cpp
  LinkingContext.cpp
  PassManager.cpp
  Resolver.cpp
  SymbolTable.cpp
  )

target_link_libraries(lldCore ${cmake_2_8_12_INTERFACE}
  lldNative
  lldYAML
  )
