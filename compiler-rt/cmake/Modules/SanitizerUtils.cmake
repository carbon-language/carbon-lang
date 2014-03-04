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
  set(stamp ${CMAKE_CURRENT_BINARY_DIR}/${name}.syms-stamp)
  add_custom_command(OUTPUT ${stamp}
    COMMAND ${PYTHON_EXECUTABLE}
      ${SANITIZER_GEN_DYNAMIC_LIST} $<TARGET_FILE:${name}> ${ARGN}
      > $<TARGET_FILE:${name}>.syms
    COMMAND ${CMAKE_COMMAND} -E touch ${stamp}
    DEPENDS ${name} ${SANITIZER_GEN_DYNAMIC_LIST} ${ARGN}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating exported symbols for ${name}"
    VERBATIM)
  add_custom_target(${name}-symbols ALL
    DEPENDS ${stamp}
    SOURCES ${SANITIZER_GEN_DYNAMIC_LIST} ${ARGN})

  if(NOT CMAKE_VERSION VERSION_LESS 3.0)
    install(FILES $<TARGET_FILE:${name}>.syms
            DESTINATION ${COMPILER_RT_LIBRARY_INSTALL_DIR})
  else()
    # Per-config install location.
    if(CMAKE_CONFIGURATION_TYPES)
      foreach(c ${CMAKE_CONFIGURATION_TYPES})
        get_target_property(libfile ${name} LOCATION_${c})
        install(FILES ${libfile}.syms CONFIGURATIONS ${c}
          DESTINATION ${COMPILER_RT_LIBRARY_INSTALL_DIR})
      endforeach()
    else()
      get_target_property(libfile ${name} LOCATION_${CMAKE_BUILD_TYPE})
      install(FILES ${libfile}.syms DESTINATION ${COMPILER_RT_LIBRARY_INSTALL_DIR})
    endif()
  endif()
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

