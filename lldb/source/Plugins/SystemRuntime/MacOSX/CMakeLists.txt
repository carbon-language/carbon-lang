set(LLVM_NO_RTTI 1)

# We depend on AppleThreadPlanStepThroughObjCTrampoline.h
include_directories(../../LanguageRuntime/ObjC/AppleObjCRuntime)

add_lldb_library(lldbPluginSystemRuntimeMacOSX
  AppleGetItemInfoHandler.cpp
  AppleGetPendingItemsHandler.cpp
  AppleGetQueuesHandler.cpp
  AppleGetThreadItemInfoHandler.cpp
  SystemRuntimeMacOSX.cpp
  )
