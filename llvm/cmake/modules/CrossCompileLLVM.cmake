
if( ${LLVM_TABLEGEN} STREQUAL "tblgen" )
  # TODO: remove this when autobuilding the native tblgen works.
  message(FATAL_ERROR
    "Set LLVM_TABLEGEN to the full route to a native tblgen executable")

  message(STATUS "Configuring native TableGen...")
  set(CX_NATIVE_TG_DIR "${CMAKE_BINARY_DIR}/native")

  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${CX_NATIVE_TG_DIR}
    RESULT_VARIABLE CX_NATIVE_TG_RV)
  if( NOT CX_NATIVE_TG_RV EQUAL 0 )
    message(FATAL_ERROR "Failed to create directory ${CX_NATIVE_TG_DIR}")
  endif()

  execute_process(
    COMMAND ${CMAKE_COMMAND} -UMAKE_TOOLCHAIN_FILE -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
    WORKING_DIRECTORY ${CX_NATIVE_TG_DIR}
    RESULT_VARIABLE CX_NATIVE_TG_RV
    ERROR_VARIABLE CX_NATIVE_TG_ERROR)
  if( NOT CX_NATIVE_TG_RV EQUAL 0 )
    message(FATAL_ERROR
      "Error while configuring native TableGen:\n${CX_NATIVE_TG_ERROR}")
  endif()

  message(STATUS "Native TableGen configured.")

  set(LLVM_TABLEGEN "${CX_NATIVE_TG_DIR}/bin/tblgen")

  add_custom_command(OUTPUT ${LLVM_TABLEGEN}
    COMMAND ${CMAKE_BUILD_TOOL} -C ${CX_NATIVE_TG_DIR}/utils/TableGen
    COMMENT "Building native TableGen...")
  add_custom_target(NativeTableGen DEPENDS ${LLVM_TABLEGEN})
  add_dependencies(tblgen NativeTableGen)

  # TODO: We should clean the native build when the `clean target
  # is invoked. This doesn't work.
#   add_custom_command(TARGET clean
#     COMMAND ${CMAKE_BUILD_TOOL} -C ${CX_NATIVE_TG_DIR}/utils/TableGen clean
#     POST_BUILD
#     COMMENT "Cleaning native TableGen...")
endif()
