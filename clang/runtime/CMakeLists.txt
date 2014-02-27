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

  if (CMAKE_GENERATOR MATCHES "Make")
    # Use special command for Makefiles to support parallelism.
    set(check_command "$(MAKE)" "check-all")
  else()
    set(check_command ${CMAKE_COMMAND} --build . --target check-all
                                       --config $<CONFIGURATION>)
  endif()
  ExternalProject_Get_Property(compiler-rt BINARY_DIR)
  add_custom_target(check-compiler-rt
          COMMAND ${check_command}
          DEPENDS compiler-rt
          WORKING_DIRECTORY ${BINARY_DIR}
          VERBATIM)
  # Add binaries that compiler-rt tests depend on.
  add_dependencies(check-compiler-rt FileCheck count
                                     not llvm-nm llvm-symbolizer)
endif()
