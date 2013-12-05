set(LLVM_LINK_COMPONENTS ${LLVM_TARGETS_TO_BUILD} bitreader asmparser irreader)

add_llvm_tool(llc
  llc.cpp
  )
set_target_properties(llc PROPERTIES ENABLE_EXPORTS 1)
