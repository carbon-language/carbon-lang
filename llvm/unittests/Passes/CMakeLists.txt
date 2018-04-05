set(LLVM_LINK_COMPONENTS Support Passes Core)

add_llvm_unittest(PluginsTests PluginsTest.cpp)
export_executable_symbols(PluginsTests)

add_library(TestPlugin MODULE TestPlugin.cxx)

set_output_directory(TestPlugin
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
  LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
  )

set_target_properties(TestPlugin
  PROPERTIES PREFIX ""
  SUFFIX ".so"
  )

add_dependencies(TestPlugin intrinsics_gen)
add_dependencies(PluginsTests TestPlugin)
