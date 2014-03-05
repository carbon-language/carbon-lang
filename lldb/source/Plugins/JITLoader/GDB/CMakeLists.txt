set(LLVM_NO_RTTI 1)

include_directories(.)

add_lldb_library(lldbPluginJITLoaderGDB
  JITLoaderGDB.cpp
  )

