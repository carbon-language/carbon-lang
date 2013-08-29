include(LLVMParseArguments)

set(SANITIZER_GEN_DYNAMIC_LIST
  ${COMPILER_RT_SOURCE_DIR}/lib/sanitizer_common/scripts/gen_dynamic_list.py)

set(SANITIZER_LINT_SCRIPT
  ${COMPILER_RT_SOURCE_DIR}/lib/sanitizer_common/scripts/check_lint.sh)

# Create a target "<name>-symbols" that would generate the list of symbols
# that need to be exported from sanitizer runtime "<name>". Function
# interceptors are exported automatically, user can also provide files with
# symbol names that should be exported as well.
#   add_sanitizer_rt_symbols(<name> <files with extra symbols to export>)
macro(add_sanitizer_rt_symbols name)
  get_target_property(libfile ${name} LOCATION)
  set(symsfile "${libfile}.syms")
  add_custom_command(OUTPUT ${symsfile}
    COMMAND ${PYTHON_EXECUTABLE}
      ${SANITIZER_GEN_DYNAMIC_LIST} ${libfile} ${ARGN}
      > ${symsfile}
    DEPENDS ${name} ${SANITIZER_GEN_DYNAMIC_LIST} ${ARGN}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating exported symbols for ${name}"
    VERBATIM)
  add_custom_target(${name}-symbols ALL
    DEPENDS ${symsfile}
    SOURCES ${SANITIZER_GEN_DYNAMIC_LIST} ${ARGN})
  install(FILES ${symsfile} DESTINATION ${COMPILER_RT_LIBRARY_INSTALL_DIR})
endmacro()

# Add target to check code style for sanitizer runtimes.
if(UNIX)
  add_custom_target(SanitizerLintCheck
    COMMAND LLVM_CHECKOUT=${LLVM_MAIN_SRC_DIR} SILENT=1 TMPDIR=
      PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
      ${SANITIZER_LINT_SCRIPT}
    DEPENDS ${SANITIZER_LINT_SCRIPT}
    COMMENT "Running lint check for sanitizer sources..."
    VERBATIM)
endif()

