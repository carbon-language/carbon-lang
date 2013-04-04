set(LLVM_SOURCE_DIR "${LLVM_MAIN_SRC_DIR}")
set(LLVM_BINARY_DIR "${LLVM_BINARY_DIR}")
set(LLVM_BUILD_MODE "%(build_mode)s")
set(LLVM_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}/%(build_config)s")
set(LLVM_LIBS_DIR "${LLVM_BINARY_DIR}/lib/%(build_config)s")
set(CLANG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(CLANG_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/..")
if(BUILD_SHARED_LIBS)
  set(ENABLE_SHARED 1)
else()
  set(ENABLE_SHARED 0)
endif(BUILD_SHARED_LIBS)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)

if ( NOT LLD_BUILT_STANDALONE )

  set(LLD_TEST_DEPS
    lld-test.deps
    FileCheck not llvm-nm
    lld llvm-objdump llvm-readobj
    linker-script-test
    )
  set(LLD_TEST_PARAMS
    lld_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
    )

  add_lit_testsuite(check-lld "Running lld regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    PARAMS ${LLD_TEST_PARAMS}
    DEPENDS ${LLD_TEST_DEPS}
    )

  set_target_properties(check-lld PROPERTIES FOLDER "lld tests")

else()

  include(FindPythonInterp)
  if (PYTHONINTERP_FOUND)
    if (LLVM_MAIN_SRC_DIR)
      set(LIT "${LLVM_SOURCE_DIR}/utils/lit/lit.py")
    else()
      set(LIT "${PATH_TO_LLVM_BUILD}/bin/${CMAKE_CFG_INTDIR}/llvm-lit")
      # Installed LLVM does not contain ${CMAKE_CFG_INTDIR} in paths.
      if (NOT EXISTS ${LIT})
        set(LIT "${PATH_TO_LLVM_BUILD}/bin/llvm-lit")
      endif()
    endif()

    if (PATH_TO_LLVM_BUILD)
      set(LLD_TEST_EXTRA_ARGS "--path=${LLD_BINARY_DIR}/bin/${CMAKE_CFG_INTDIR}")
    endif()

    set(LIT_ARGS "${LLD_TEST_EXTRA_ARGS} ${LLVM_LIT_ARGS}")
    separate_arguments(LIT_ARGS)

    add_custom_target(lld-test.deps)
    set_target_properties(lld-test.deps PROPERTIES FOLDER "lld tests")

    add_custom_target(check-lld
      COMMAND ${PYTHON_EXECUTABLE}
                  ${LIT}
                  --param lld_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
                  --param build_config=${CMAKE_CFG_INTDIR}
                  --param build_mode=${RUNTIME_BUILD_MODE}
                  ${LIT_ARGS}
                  ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Running lld regression tests"
      DEPENDS lld-test.deps
      )
    set_target_properties(check-lld PROPERTIES FOLDER "lld tests")
  endif()

endif()

# Add a legacy target spelling: lld-test
add_custom_target(lld-test)
add_dependencies(lld-test check-lld)
set_target_properties(lld-test PROPERTIES FOLDER "lld tests")
