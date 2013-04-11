set(SANITIZER_HEADERS
  sanitizer/asan_interface.h
  sanitizer/common_interface_defs.h
  sanitizer/linux_syscall_hooks.h
  sanitizer/msan_interface.h)

set(output_dir ${LLVM_BINARY_DIR}/lib/clang/${CLANG_VERSION}/include)

if(MSVC_IDE OR XCODE)
   set(other_output_dir ${LLVM_BINARY_DIR}/bin/lib/clang/${CLANG_VERSION}/include)
endif()

# Copy compiler-rt headers to the build tree.
set(out_files)
foreach( f ${SANITIZER_HEADERS} )
  set( src ${CMAKE_CURRENT_SOURCE_DIR}/${f} )
  set( dst ${output_dir}/${f} )
  add_custom_command(OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying compiler-rt's ${f}...")
  list(APPEND out_files ${dst})

  if(other_output_dir)
   set(other_dst ${other_output_dir}/${f})
    add_custom_command(OUTPUT ${other_dst}
      DEPENDS ${src}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${other_dst}
      COMMENT "Copying compiler-rt's ${f}...")    
    list(APPEND out_files ${other_dst})
  endif()
endforeach( f )

add_custom_target(compiler-rt-headers ALL DEPENDS ${out_files})

# Install sanitizer headers.
install(FILES ${SANITIZER_HEADERS}
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION ${LIBCLANG_INSTALL_PATH}/include/sanitizer)
