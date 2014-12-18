# TODO: Set the install directory.

include(ExternalProject)

set(known_subdirs
  "libcxx"
  )

foreach (dir ${known_subdirs})
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${dir})
  endif()
endforeach()

function(get_ext_project_build_command out_var target)
  if (CMAKE_GENERATOR MATCHES "Make")
    # Use special command for Makefiles to support parallelism.
    set(${out_var} "$(MAKE)" "${target}" PARENT_SCOPE)
  else()
    set(${out_var} ${CMAKE_COMMAND} --build . --target ${target}
                                    --config $<CONFIGURATION> PARENT_SCOPE)
  endif()
endfunction()

set(COMPILER_RT_SRC_ROOT ${LLVM_MAIN_SRC_DIR}/projects/compiler-rt)
if(LLVM_BUILD_EXTERNAL_COMPILER_RT AND EXISTS ${COMPILER_RT_SRC_ROOT}/)
  if(CMAKE_GENERATOR MATCHES "Ninja")
    message(FATAL_ERROR
            "Ninja generator can't build compiler-rt as ExternalProject."
            "Unset LLVM_BUILD_EXTERNAL_COMPILER_RT, or don't use Ninja."
            "See http://www.cmake.org/Bug/view.php?id=14771")
  endif()

  # Add compiler-rt as an external project.
  set(COMPILER_RT_PREFIX ${CMAKE_BINARY_DIR}/projects/compiler-rt)
  
  ExternalProject_Add(compiler-rt
    PREFIX ${COMPILER_RT_PREFIX}
    SOURCE_DIR ${COMPILER_RT_SRC_ROOT}
    CMAKE_ARGS -DCMAKE_C_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang
               -DCMAKE_CXX_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang++
               -DCMAKE_BUILD_TYPE=Release
               -DLLVM_CONFIG_PATH=${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-config
               -DCOMPILER_RT_OUTPUT_DIR=${LLVM_LIBRARY_OUTPUT_INTDIR}/clang/${CLANG_VERSION}
               -DCOMPILER_RT_EXEC_OUTPUT_DIR=${LLVM_RUNTIME_OUTPUT_INTDIR}
               -DCOMPILER_RT_INSTALL_PATH=lib${LLVM_LIBDIR_SUFFIX}/clang/${CLANG_VERSION}
               -DCOMPILER_RT_INCLUDE_TESTS=${LLVM_INCLUDE_TESTS}
               -DCOMPILER_RT_ENABLE_WERROR=ON
    INSTALL_COMMAND ""
    STEP_TARGETS configure build
    )
  # Due to a bug, DEPENDS in ExternalProject_Add doesn't work
  # in CMake 2.8.9 and 2.8.10.
  add_dependencies(compiler-rt llvm-config clang)

  # Add a custom step to always re-configure compiler-rt (in case some of its
  # sources have changed).
  ExternalProject_Add_Step(compiler-rt force-reconfigure
    DEPENDERS configure
    ALWAYS 1
    )

  ExternalProject_Add_Step(compiler-rt clobber
    COMMAND ${CMAKE_COMMAND} -E remove_directory <BINARY_DIR>
    COMMAND ${CMAKE_COMMAND} -E make_directory <BINARY_DIR>
    COMMENT "Clobberring compiler-rt build directory..."
    DEPENDERS configure
    DEPENDS ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang
  )

  ExternalProject_Get_Property(compiler-rt BINARY_DIR)
  set(COMPILER_RT_BINARY_DIR ${BINARY_DIR})

  # Add top-level targets that build specific compiler-rt runtimes.
  set(COMPILER_RT_RUNTIMES asan builtins dfsan lsan msan profile tsan ubsan)
  foreach(runtime ${COMPILER_RT_RUNTIMES})
    get_ext_project_build_command(build_runtime_cmd ${runtime})
    add_custom_target(${runtime}
      COMMAND ${build_runtime_cmd}
      DEPENDS compiler-rt-configure
      WORKING_DIRECTORY ${COMPILER_RT_BINARY_DIR}
      VERBATIM)
  endforeach()

  # Add binaries that compiler-rt tests depend on.
  set(COMPILER_RT_TEST_DEPENDENCIES
    FileCheck count not llvm-nm llvm-symbolizer)

  # Add top-level targets for various compiler-rt test suites.
  set(COMPILER_RT_TEST_SUITES check-asan check-asan-dynamic check-dfsan
    check-lsan check-msan check-sanitizer check-tsan check-ubsan)
  foreach(test_suite ${COMPILER_RT_TEST_SUITES})
    get_ext_project_build_command(run_test_suite ${test_suite})
    add_custom_target(${test_suite}
      COMMAND ${run_test_suite}
      DEPENDS compiler-rt-build ${COMPILER_RT_TEST_DEPENDENCIES}
      WORKING_DIRECTORY ${COMPILER_RT_BINARY_DIR}
      VERBATIM)
  endforeach()

  # Add special target to run all compiler-rt test suites.
  get_ext_project_build_command(run_check_compiler_rt check-all)
  add_custom_target(check-compiler-rt
    COMMAND ${run_check_compiler_rt}
    DEPENDS compiler-rt-build ${COMPILER_RT_TEST_DEPENDENCIES}
    WORKING_DIRECTORY ${COMPILER_RT_BINARY_DIR}
    VERBATIM)
endif()
