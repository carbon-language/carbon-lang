if(LLVM_TARGETS_TO_BUILD MATCHES "X86")
  add_subdirectory(
    llvm-cfi-verify
  )
  add_subdirectory(
    llvm-exegesis
  )
endif()

