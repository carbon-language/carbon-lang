set(LLVM_LINK_COMPONENTS
  Object
  Support
  )

add_llvm_unittest(ArchiveFileDescriptor
  ArchiveFileDescriptor.cpp
  )
