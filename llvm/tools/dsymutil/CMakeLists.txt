set(LLVM_LINK_COMPONENTS
  DebugInfo
  Object
  Support
  )

add_llvm_tool(llvm-dsymutil
  dsymutil.cpp
  BinaryHolder.cpp
  DebugMap.cpp
  DwarfLinker.cpp
  MachODebugMapParser.cpp
  )

