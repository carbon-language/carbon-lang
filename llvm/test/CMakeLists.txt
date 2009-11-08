foreach(c ${LLVM_TARGETS_TO_BUILD})
  set(TARGETS_BUILT "${TARGETS_BUILT} ${c}")
endforeach(c)
set(TARGETS_TO_BUILD ${TARGETS_BUILT})

include(FindPythonInterp)
if(PYTHONINTERP_FOUND)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/site.exp.in
    ${CMAKE_CURRENT_BINARY_DIR}/site.exp)

  MAKE_DIRECTORY(${CMAKE_CURRENT_BINARY_DIR}/Unit)

  add_custom_target(check
    COMMAND sed -e "s#\@LLVM_SOURCE_DIR\@#${LLVM_MAIN_SRC_DIR}#"
                -e "s#\@LLVM_BINARY_DIR\@#${LLVM_BINARY_DIR}#"
                -e "s#\@LLVM_TOOLS_DIR\@#${LLVM_TOOLS_BINARY_DIR}/${CMAKE_CFG_INTDIR}#"
                -e "s#\@LLVMGCCDIR\@##"
                ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in >
                ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
    COMMAND sed -e "s#\@LLVM_SOURCE_DIR\@#${LLVM_MAIN_SRC_DIR}#"
                -e "s#\@LLVM_BINARY_DIR\@#${LLVM_BINARY_DIR}#"
                -e "s#\@LLVM_TOOLS_DIR\@#${LLVM_TOOLS_BINARY_DIR}/${CMAKE_CFG_INTDIR}#"
                -e "s#\@LLVMGCCDIR\@##"
                -e "s#\@LLVM_BUILD_MODE\@#${CMAKE_CFG_INTDIR}#"
                ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in >
                ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
    COMMAND ${PYTHON_EXECUTABLE}
                ${LLVM_SOURCE_DIR}/utils/lit/lit.py
                -sv
                ${CMAKE_CURRENT_BINARY_DIR}
                DEPENDS
                COMMENT "Running LLVM regression tests")

endif()
