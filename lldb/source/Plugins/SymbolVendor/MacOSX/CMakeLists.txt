include_directories(${LIBXML2_INCLUDE_DIR})

add_lldb_library(lldbPluginSymbolVendorMacOSX
  SymbolVendorMacOSX.cpp
  )
