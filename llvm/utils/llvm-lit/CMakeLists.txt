if (WIN32 AND NOT CYGWIN)
  # llvm-lit needs suffix.py for multiprocess to find a main module.
  set(suffix .py)
endif ()

configure_file(
  llvm-lit.in
  ${LLVM_TOOLS_BINARY_DIR}/llvm-lit${suffix}
  )
