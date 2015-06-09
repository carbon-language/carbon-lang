set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  LibDriver
  Object
  Support
  )

add_llvm_tool(llvm-ar
  llvm-ar.cpp
  )

if(UNIX)
  set(LLVM_LINK_OR_COPY create_symlink)
  set(llvm_ar_binary "llvm-ar${CMAKE_EXECUTABLE_SUFFIX}")
else()
  set(LLVM_LINK_OR_COPY copy)
  set(llvm_ar_binary "${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-ar${CMAKE_EXECUTABLE_SUFFIX}")
endif()

set(llvm_ranlib "${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-ranlib${CMAKE_EXECUTABLE_SUFFIX}")

add_custom_command(OUTPUT ${llvm_ranlib}
                   COMMAND ${CMAKE_COMMAND} -E ${LLVM_LINK_OR_COPY} "${llvm_ar_binary}" "${llvm_ranlib}"
                   DEPENDS llvm-ar)

add_custom_target(llvm-ranlib ALL DEPENDS ${llvm_ranlib})
set_target_properties(llvm-ranlib PROPERTIES FOLDER Tools)

set(llvm_lib "${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-lib${CMAKE_EXECUTABLE_SUFFIX}")

add_custom_command(OUTPUT ${llvm_lib}
                   COMMAND ${CMAKE_COMMAND} -E ${LLVM_LINK_OR_COPY} "${llvm_ar_binary}" "${llvm_lib}"
                   DEPENDS llvm-ar)

add_custom_target(llvm-lib ALL DEPENDS ${llvm_lib})
set_target_properties(llvm-lib PROPERTIES FOLDER Tools)

install(SCRIPT install_symlink.cmake -DCMAKE_INSTALL_PREFIX=\"${CMAKE_INSTALL_PREFIX}\")
