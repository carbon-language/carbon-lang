
if( ${LLVM_TABLEGEN} STREQUAL "tblgen" )
  set(LLVM_TABLEGEN "${CX_NATIVE_TG_DIR}/bin/tblgen")

  message(STATUS "CX_NATIVE_TG_DIR : ${CX_NATIVE_TG_DIR}")
  add_custom_command(OUTPUT ${CX_NATIVE_TG_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CX_NATIVE_TG_DIR}
    COMMENT "Creating ${CX_NATIVE_TG_DIR}...")

  add_custom_command(OUTPUT ${CX_NATIVE_TG_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -UMAKE_TOOLCHAIN_FILE -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
    WORKING_DIRECTORY ${CX_NATIVE_TG_DIR}
    DEPENDS ${CX_NATIVE_TG_DIR}
    COMMENT "Configuring native TableGen...")

  add_custom_command(OUTPUT ${LLVM_TABLEGEN}
    COMMAND ${CMAKE_BUILD_TOOL}
    DEPENDS ${CX_NATIVE_TG_DIR}/CMakeCache.txt
    WORKING_DIRECTORY ${CX_NATIVE_TG_DIR}/utils/TableGen
    COMMENT "Building native TableGen...")
  add_custom_target(NativeTableGen DEPENDS ${LLVM_TABLEGEN})

  add_dependencies(tblgen NativeTableGen)

  # TODO: We should clean the native build when the `clean target
  # is invoked. This doesn't work.
  # add_custom_command(TARGET clean
  #   COMMAND ${CMAKE_BUILD_TOOL} -C ${CX_NATIVE_TG_DIR}/utils/TableGen clean
  #   POST_BUILD
  #   COMMENT "Cleaning native TableGen...")
endif()
