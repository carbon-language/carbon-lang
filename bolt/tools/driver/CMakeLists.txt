set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  BOLTProfile
  BOLTRewrite
  BOLTUtils
  MC
  Object
  Support
  )

add_llvm_tool(llvm-bolt
  llvm-bolt.cpp

  DEPENDS
  bolt_rt
  )

add_llvm_tool_symlink(perf2bolt llvm-bolt)
add_llvm_tool_symlink(llvm-boltdiff llvm-bolt)
add_llvm_tool_symlink(llvm-bolt-heatmap llvm-bolt)

include_directories( ${BOLT_SOURCE_DIR}/lib )
