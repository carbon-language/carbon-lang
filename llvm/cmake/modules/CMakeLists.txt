set(llvm_cmake_builddir "${LLVM_BINARY_DIR}/share/llvm/cmake")

configure_file(
  LLVM.cmake
  ${llvm_cmake_builddir}/LLVM.cmake)

install(FILES
  ${llvm_cmake_builddir}/LLVM.cmake
  LLVMConfig.cmake
  LLVMLibDeps.cmake
  DESTINATION share/llvm/cmake)
