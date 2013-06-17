set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  bitreader
  object)

add_llvm_tool(llvm-readobj
  llvm-readobj.cpp
  ObjDumper.cpp
  COFFDumper.cpp
  ELFDumper.cpp
  MachODumper.cpp
  Error.cpp
  StreamWriter.cpp
  )
