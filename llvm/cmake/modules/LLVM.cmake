set(LLVM_COMMON_DEPENDS @LLVM_COMMON_DEPENDS@)

set(llvm_libs @llvm_libs@)

set(llvm_lib_targets @llvm_lib_targets@)

set(LLVM_TARGETS_TO_BUILD @LLVM_TARGETS_TO_BUILD@)

set(LLVM_TOOLS_BINARY_DIR @LLVM_TOOLS_BINARY_DIR@)

set(LLVM_ENABLE_THREADS @LLVM_ENABLE_THREADS@)

if( NOT EXISTS LLVMConfig.cmake )
  set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    "@LLVM_SOURCE_DIR@/cmake/modules")
endif()

include( LLVMConfig )
