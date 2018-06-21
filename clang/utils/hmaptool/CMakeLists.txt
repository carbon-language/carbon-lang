set(CLANG_HMAPTOOL hmaptool)

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/bin/${CLANG_HMAPTOOL}
                   COMMAND ${CMAKE_COMMAND} -E make_directory
                     ${CMAKE_BINARY_DIR}/bin
                   COMMAND ${CMAKE_COMMAND} -E copy
                     ${CMAKE_CURRENT_SOURCE_DIR}/${CLANG_HMAPTOOL}
                     ${CMAKE_BINARY_DIR}/bin/
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${CLANG_HMAPTOOL})

list(APPEND Depends ${CMAKE_BINARY_DIR}/bin/${CLANG_HMAPTOOL})
install(PROGRAMS ${CLANG_HMAPTOOL} DESTINATION bin)

add_custom_target(hmaptool ALL DEPENDS ${Depends})
set_target_properties(hmaptool PROPERTIES FOLDER "Utils")

