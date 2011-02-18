set(llvm_cmake_builddir "${LLVM_BINARY_DIR}/share/llvm/cmake")

get_property(llvm_libs GLOBAL PROPERTY LLVM_LIBS)
get_property(llvm_lib_targets GLOBAL PROPERTY LLVM_LIB_TARGETS)

configure_file(
  LLVM.cmake
  ${llvm_cmake_builddir}/LLVM.cmake
  @ONLY)

install(FILES
  ${llvm_cmake_builddir}/LLVM.cmake
  LLVMConfig.cmake
  LLVMLibDeps.cmake
  DESTINATION share/llvm/cmake)

install(DIRECTORY .
  DESTINATION share/llvm/cmake
  FILES_MATCHING PATTERN *.cmake
  PATTERN .svn EXCLUDE
  PATTERN LLVM.cmake EXCLUDE
  PATTERN LLVMConfig.cmake EXCLUDE
  PATTERN LLVMLibDeps.cmake EXCLUDE
  PATTERN FindBison.cmake EXCLUDE
  PATTERN GetTargetTriple.cmake EXCLUDE
  PATTERN VersionFromVCS.cmake EXCLUDE
  PATTERN CheckAtomic.cmake EXCLUDE)

install(FILES
  ${llvm_cmake_builddir}/LLVM.cmake
  LLVMConfig.cmake
  LLVMLibDeps.cmake
  DESTINATION share/llvm/cmake)
