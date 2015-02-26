set(LLVM_LINK_COMPONENTS
  Support
  DebugInfoPDB
  )

add_llvm_tool(llvm-pdbdump
  llvm-pdbdump.cpp
  BuiltinDumper.cpp
  ClassDefinitionDumper.cpp
  CompilandDumper.cpp
  FunctionDumper.cpp
  TypeDumper.cpp
  TypedefDumper.cpp
  VariableDumper.cpp
  )
