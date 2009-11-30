set(LLVM_NO_RTTI 1)

set( LLVM_USED_LIBS
  clangDriver
  clangBasic
  )

set(LLVM_LINK_COMPONENTS system support bitreader bitwriter)

add_clang_executable(clang
  driver.cpp
  cc1_main.cpp
  )

add_dependencies(clang clang-cc)

if(UNIX)
  set(CLANGXX_LINK_OR_COPY create_symlink)
else()
  set(CLANGXX_LINK_OR_COPY copy)
endif()

# Create the clang++ symlink in the build directory.
add_custom_target(clang++ ALL
  ${CMAKE_COMMAND} -E ${CLANGXX_LINK_OR_COPY}
    "${LLVM_BINARY_DIR}/bin/${CMAKE_CFG_INTDIR}/clang${CMAKE_EXECUTABLE_SUFFIX}"
    "${LLVM_BINARY_DIR}/bin/${CMAKE_CFG_INTDIR}/clang++${CMAKE_EXECUTABLE_SUFFIX}"
  DEPENDS clang)

install(TARGETS clang
  RUNTIME DESTINATION bin)

# Create the clang++ symlink at installation time.
install(CODE "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E ${CLANGXX_LINK_OR_COPY} \"${CMAKE_INSTALL_PREFIX}/bin/clang${CMAKE_EXECUTABLE_SUFFIX}\" \"${CMAKE_INSTALL_PREFIX}/bin/clang++${CMAKE_EXECUTABLE_SUFFIX}\")")
