add_llvm_executable(lld
  lld.cpp
  )

target_link_libraries(lld
  lldDriver
  lldCOFF
  LLVMSupport
  )

install(TARGETS lld
  RUNTIME DESTINATION bin)

if(UNIX)
  set(LLD_LINK_OR_COPY create_symlink)
  set(lld_binary "lld${CMAKE_EXECUTABLE_SUFFIX}")

else()
  set(LLD_LINK_OR_COPY copy)
  set(lld_binary "${LLVM_RUNTIME_OUTPUT_INTDIR}/lld${CMAKE_EXECUTABLE_SUFFIX}")
endif()

set(lld_link "${LLVM_RUNTIME_OUTPUT_INTDIR}/lld-link${CMAKE_EXECUTABLE_SUFFIX}")

add_custom_command(TARGET lld POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E ${LLD_LINK_OR_COPY} "${lld_binary}" "${lld_link}")

install(SCRIPT install_symlink.cmake -DCMAKE_INSTALL_PREFIX=\"${CMAKE_INSTALL_PREFIX}\")
