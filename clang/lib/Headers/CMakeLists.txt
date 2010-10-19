set(files
  altivec.h
  avxintrin.h
  emmintrin.h	
  float.h		
  immintrin.h
  iso646.h	
  limits.h	
  mm_malloc.h	
  mmintrin.h	
  pmmintrin.h	
  smmintrin.h
  stdarg.h	
  stdbool.h	
  stddef.h	
  stdint.h	
  tgmath.h
  tmmintrin.h
  xmmintrin.h)

set(output_dir ${LLVM_BINARY_DIR}/lib/clang/${CLANG_VERSION}/include)

# If we are in an IDE that has a configuration directory, we need to
# create a second copy of the headers so that 'clang' can find them if
# it's run from the build directory.
if(MSVC_IDE OR XCODE)
   set(other_output_dir ${LLVM_BINARY_DIR}/bin/lib/clang/${CLANG_VERSION}/include)
endif()

# Generate arm_neon.h
set(LLVM_TARGET_DEFINITIONS ${CLANG_SOURCE_DIR}/include/clang/Basic/arm_neon.td)
tablegen(arm_neon.h.inc -gen-arm-neon)

set(out_files)
foreach( f ${files} )
  set( src ${CMAKE_CURRENT_SOURCE_DIR}/${f} )
  set( dst ${output_dir}/${f} )
  add_custom_command(OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying clang's ${f}...")
  list(APPEND out_files ${dst})

  if(other_output_dir)
   set(other_dst ${other_output_dir}/${f})
    add_custom_command(OUTPUT ${other_dst}
      DEPENDS ${src}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${other_dst}
      COMMENT "Copying clang's ${f}...")    
    list(APPEND out_files ${other_dst})
  endif()
endforeach( f )

add_custom_command(OUTPUT ${output_dir}/arm_neon.h 
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/arm_neon.h.inc
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/arm_neon.h.inc ${output_dir}/arm_neon.h
  COMMENT "Copying clang's arm_neon.h...")
list(APPEND out_files ${output_dir}/arm_neon.h)

if (other_output_dir)
    set(other_dst ${other_output_dir}/arm_neon.h)
    add_custom_command(OUTPUT ${other_dst}
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/arm_neon.h.inc
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/arm_neon.h.inc ${other_dst}
      COMMENT "Copying clang's arm_neon.h...")
    list(APPEND out_files ${other_dst})
endif ()

add_custom_target(clang-headers ALL DEPENDS ${out_files})

install(FILES ${files} ${output_dir}/arm_neon.h
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION lib${LLVM_LIBDIR_SUFFIX}/clang/${CLANG_VERSION}/include)
