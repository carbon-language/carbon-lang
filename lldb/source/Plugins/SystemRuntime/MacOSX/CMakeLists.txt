set(LLVM_NO_RTTI 1)

add_lldb_library(lldbPluginSystemRuntimeMacOSX
  AppleGetItemInfoHandler.cpp
  AppleGetPendingItemsHandler.cpp
  AppleGetQueuesHandler.cpp
  AppleGetThreadItemInfoHandler.cpp
  SystemRuntimeMacOSX.cpp
  )
