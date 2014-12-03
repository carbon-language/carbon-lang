set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  Object
  Support
  )

add_llvm_tool(llvm-ar
  llvm-ar.cpp
  )

if(UNIX)
  set(LLVM_LINK_OR_COPY create_symlink)
else()
  set(LLVM_LINK_OR_COPY copy)
endif()

set(llvm_ar_binary "llvm-ar${CMAKE_EXECUTABLE_SUFFIX}")
set(llvm_ranlib "${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-ranlib${CMAKE_EXECUTABLE_SUFFIX}")

add_custom_command(OUTPUT ${llvm_ranlib}
                   COMMAND ${CMAKE_COMMAND} -E ${LLVM_LINK_OR_COPY} "${llvm_ar_binary}" "${llvm_ranlib}")

add_custom_target(llvm_ranlib_build ALL DEPENDS ${llvm_ranlib})
install(FILES "${llvm_ranlib}" DESTINATION bin)
