foreach(c ${LLVM_TARGETS_TO_BUILD})
  set(TARGETS_BUILT "${TARGETS_BUILT} ${c}")
endforeach(c)
set(TARGETS_TO_BUILD ${TARGETS_BUILT})

include(FindPythonInterp)
if(PYTHONINTERP_FOUND)
  get_target_property(LLVM_TOOLS_PATH llvm-config RUNTIME_OUTPUT_DIRECTORY)

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/site.exp.in
    ${CMAKE_CURRENT_BINARY_DIR}/site.exp)

  add_custom_target(llvm-test
    COMMAND sed -e "s#\@LLVM_SOURCE_DIR\@#${LLVM_MAIN_SRC_DIR}#"
                -e "s#\@LLVM_BINARY_DIR\@#${LLVM_BINARY_DIR}#"
                -e "s#\@LLVM_TOOLS_DIR\@#${LLVM_TOOLS_PATH}/${CMAKE_CFG_INTDIR}#"
                -e "s#\@LLVMGCC_DIR\@##"
                ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in >
                ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
    COMMAND sed -e "s#\@LLVM_SOURCE_DIR\@#${LLVM_MAIN_SRC_DIR}#"
                -e "s#\@LLVM_BINARY_DIR\@#${LLVM_BINARY_DIR}#"
                -e "s#\@LLVM_TOOLS_DIR\@#${LLVM_TOOLS_PATH}/${CMAKE_CFG_INTDIR}#"
                -e "s#\@LLVMGCC_DIR\@##"
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
