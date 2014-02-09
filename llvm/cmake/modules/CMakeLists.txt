set(LLVM_INSTALL_PACKAGE_DIR share/llvm/cmake)
set(llvm_cmake_builddir "${LLVM_BINARY_DIR}/${LLVM_INSTALL_PACKAGE_DIR}")
set(LLVM_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

get_property(llvm_libs GLOBAL PROPERTY LLVM_LIBS)

foreach(lib ${llvm_libs})
  get_property(llvm_lib_deps GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_${lib})
  set(all_llvm_lib_deps
    "${all_llvm_lib_deps}\nset_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_${lib} ${llvm_lib_deps})")
endforeach(lib)

configure_file(
  LLVMConfig.cmake.in
  ${llvm_cmake_builddir}/LLVMConfig.cmake
  @ONLY)

configure_file(
  LLVMConfigVersion.cmake.in
  ${llvm_cmake_builddir}/LLVMConfigVersion.cmake
  @ONLY)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(FILES
    ${llvm_cmake_builddir}/LLVMConfig.cmake
    ${llvm_cmake_builddir}/LLVMConfigVersion.cmake
    LLVM-Config.cmake
    DESTINATION ${LLVM_INSTALL_PACKAGE_DIR})

  install(DIRECTORY .
    DESTINATION ${LLVM_INSTALL_PACKAGE_DIR}
    FILES_MATCHING PATTERN *.cmake
    PATTERN .svn EXCLUDE
    PATTERN LLVMConfig.cmake EXCLUDE
    PATTERN LLVMConfigVersion.cmake EXCLUDE
    PATTERN LLVM-Config.cmake EXCLUDE
    PATTERN GetHostTriple.cmake EXCLUDE
    PATTERN VersionFromVCS.cmake EXCLUDE
    PATTERN CheckAtomic.cmake EXCLUDE)
endif()
