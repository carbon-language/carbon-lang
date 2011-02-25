set( LLVM_USED_LIBS
  clangAST
  clangAnalysis
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangFrontendTool
  clangIndex
  clangLex
  clangParse
  clangRewrite
  clangSema
  clangSerialization
  clangStaticAnalyzerFrontend
  clangStaticAnalyzerCheckers
  clangStaticAnalyzerCore
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

set_target_properties(clang PROPERTIES VERSION ${CLANG_EXECUTABLE_VERSION})

if(UNIX)
  set(CLANGXX_LINK_OR_COPY create_symlink)
else()
  set(CLANGXX_LINK_OR_COPY copy)
endif()

# Create the clang++ symlink in the build directory.
set(clang_pp "${LLVM_BINARY_DIR}/bin/${CMAKE_CFG_INTDIR}/clang++${CMAKE_EXECUTABLE_SUFFIX}")
add_custom_target(clang++ ALL
  ${CMAKE_COMMAND} -E ${CLANGXX_LINK_OR_COPY}
    "${LLVM_BINARY_DIR}/bin/${CMAKE_CFG_INTDIR}/clang${CMAKE_EXECUTABLE_SUFFIX}"
    "${clang_pp}"
  DEPENDS clang)
set_target_properties(clang++ PROPERTIES FOLDER "Clang executables")

set_property(DIRECTORY APPEND
  PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${clang_pp})

install(TARGETS clang
  RUNTIME DESTINATION bin)

# Create the clang++ symlink at installation time.
install(SCRIPT clang_symlink.cmake -DCMAKE_INSTALL_PREFIX=\"${CMAKE_INSTALL_PREFIX}\")
