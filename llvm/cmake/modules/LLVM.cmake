# This file provides information and services to the final user.

set(LLVM_PACKAGE_VERSION @PACKAGE_VERSION@)

set(LLVM_COMMON_DEPENDS @LLVM_COMMON_DEPENDS@)

set(llvm_libs @llvm_libs@)

set(llvm_lib_targets @llvm_lib_targets@)

set(LLVM_ALL_TARGETS @LLVM_ALL_TARGETS@)

set(LLVM_TARGETS_TO_BUILD @LLVM_TARGETS_TO_BUILD@)

set(LLVM_TOOLS_BINARY_DIR @LLVM_TOOLS_BINARY_DIR@)

set(LLVM_ENABLE_THREADS @LLVM_ENABLE_THREADS@)

set(LLVM_NATIVE_ARCH @LLVM_NATIVE_ARCH@)

set(LLVM_ENABLE_PIC @LLVM_ENABLE_PIC@)

set(LLVM_ENABLE_THREADS @LLVM_ENABLE_THREADS)

set(HAVE_LIBDL @HAVE_LIBDL@)
set(HAVE_LIBPTHREAD @HAVE_LIBPTHREAD)

# We try to include using the current setting of CMAKE_MODULE_PATH,
# which suppossedly was filled by the user with the directory where
# this file was installed:
include( LLVMConfig OPTIONAL RESULT_VARIABLE LLVMCONFIG_INCLUDED )

# If failed, we assume that this is an un-installed build:
if( NOT LLVMCONFIG_INCLUDED )
  set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    "@LLVM_SOURCE_DIR@/cmake/modules")
  include( LLVMConfig )
endif()

