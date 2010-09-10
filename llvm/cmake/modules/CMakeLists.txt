set(LLVM_CFLAGS "${CMAKE_C_FLAGS_${uppercase_CMAKE_BUILD_TYPE}} ${LLVM_DEFINITIONS}")
set(LLVM_CXXFLAGS "${CMAKE_CXX_FLAGS_${uppercase_CMAKE_BUILD_TYPE}} ${LLVM_DEFINITIONS}")
set(LLVM_CPPFLAGS "${CMAKE_CPP_FLAGS_${uppercase_CMAKE_BUILD_TYPE}} ${LLVM_DEFINITIONS}")
llvm_get_target_libraries(LLVM_JIT_LIBS jit)
llvm_get_target_libraries(LLVM_INTERPRETER_LIBS interpreter)
llvm_get_target_libraries(LLVM_NATIVECODEGEN_LIBS nativecodegen)
get_system_libs(LLVM_SYSTEM_LIBS)

configure_file(
  LLVMConfig.cmake.in
  ${LLVM_BINARY_DIR}/cmake/modules/LLVMConfig.cmake
  @ONLY
  )

install(FILES
  ${LLVM_BINARY_DIR}/cmake/modules/LLVMConfig.cmake
  DESTINATION lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm/
  )
