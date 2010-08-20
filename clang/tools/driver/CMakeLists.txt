set(LLVM_NO_RTTI 1)

set( LLVM_USED_LIBS
  clangFrontend
  clangDriver
  clangSerialization
  clangCodeGen
  clangParse
  clangSema
  clangChecker
  clangAnalysis
  clangIndex
  clangRewrite
  clangAST
  clangLex
  clangBasic
  )

set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  bitreader
  bitwriter
  codegen
  ipo
  selectiondag
  )

add_clang_executable(clang
  driver.cpp
  cc1_main.cpp
  cc1as_main.cpp
  )

if(UNIX)
  set(CLANGXX_LINK_OR_COPY create_symlink)
  set(CLANGXX_DESTDIR $ENV{DESTDIR}/)
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
install(CODE "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E ${CLANGXX_LINK_OR_COPY} \"${CMAKE_INSTALL_PREFIX}/bin/clang${CMAKE_EXECUTABLE_SUFFIX}\" \"${CLANGXX_DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/clang++${CMAKE_EXECUTABLE_SUFFIX}\")")
