set(LLVM_LINK_COMPONENTS ${LLVM_TARGETS_TO_BUILD} bitreader asmprinter)

add_llvm_tool(llc
  llc.cpp
  )
