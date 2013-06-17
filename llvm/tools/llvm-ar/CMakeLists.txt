set(LLVM_LINK_COMPONENTS support bitreader)

add_llvm_tool(llvm-ar
  llvm-ar.cpp
  ArchiveWriter.cpp
  ArchiveReader.cpp
  Archive.cpp
  )

# TODO: Support check-local.
