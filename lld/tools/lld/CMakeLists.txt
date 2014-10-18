add_lld_executable(lld
  lld.cpp
  )

target_link_libraries(lld
  lldDriver
  )

install(TARGETS lld
  RUNTIME DESTINATION bin)

# Create the lld-link[.exe] symlink in the build directory. If symlink is not
# supported by the operating system, create a copy instead.
if(UNIX)
  set(command create_symlink)
  # Make relative symlink
  set(src "lld${CMAKE_EXECUTABLE_SUFFIX}")
else()
  set(command copy)
  set(src "${LLVM_RUNTIME_OUTPUT_INTDIR}/lld${CMAKE_EXECUTABLE_SUFFIX}")
endif()
set(dst "${LLVM_RUNTIME_OUTPUT_INTDIR}/lld-link${CMAKE_EXECUTABLE_SUFFIX}")
add_custom_command(TARGET lld POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E ${command} ${src} ${dst})
