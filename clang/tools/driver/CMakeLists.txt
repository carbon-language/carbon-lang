set(LLVM_NO_RTTI 1)

set( LLVM_USED_LIBS
  clangDriver
  clangBasic
  clangFrontend
  )

set(LLVM_LINK_COMPONENTS system support bitreader bitwriter)

add_clang_executable(clang
  driver.cpp
  cc1_main.cpp
  )

add_dependencies(clang clang-cc)

# Create the clang++ symlink in the build directory.
add_custom_target(clang++ ALL
  ${CMAKE_COMMAND} -E create_symlink 
    "${LLVM_BINARY_DIR}/bin/${CMAKE_CFG_INTDIR}/clang${CMAKE_EXECUTABLE_SUFFIX}"
    "${LLVM_BINARY_DIR}/bin/${CMAKE_CFG_INTDIR}/clang++${CMAKE_EXECUTABLE_SUFFIX}"
  DEPENDS clang)

install(TARGETS clang
  RUNTIME DESTINATION bin)

# Create the clang++ symlink at installation time.
install(CODE "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E create_symlink \"${CMAKE_INSTALL_PREFIX}/bin/clang${CMAKE_EXECUTABLE_SUFFIX}\" \"${CMAKE_INSTALL_PREFIX}/bin/clang++${CMAKE_EXECUTABLE_SUFFIX}\")")
