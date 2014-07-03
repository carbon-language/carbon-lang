set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Object
  Support
  )

add_llvm_tool(llvm-ar
  llvm-ar.cpp
  )

# FIXME: this is duplicated from the clang CMakeLists.txt
# FIXME: bin/llvm-ranlib is not a valid build target with this setup (pr17024)

if(UNIX)
  set(LLVM_LINK_OR_COPY create_symlink)
  set(llvm_ar_binary "llvm-ar${CMAKE_EXECUTABLE_SUFFIX}")
else()
  set(LLVM_LINK_OR_COPY copy)
  set(llvm_ar_binary "${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-ar${CMAKE_EXECUTABLE_SUFFIX}")
endif()

set(llvm_ranlib "${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-ranlib${CMAKE_EXECUTABLE_SUFFIX}")
add_custom_command(TARGET llvm-ar POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E ${LLVM_LINK_OR_COPY} "${llvm_ar_binary}" "${llvm_ranlib}")

set_property(DIRECTORY APPEND
  PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${llvm_ranlib})

# TODO: Support check-local.
