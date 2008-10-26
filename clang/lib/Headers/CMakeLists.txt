set(files
  iso646.h
  mmintrin.h
  stdarg.h
  stdbool.h
  stddef.h
  )

set(output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/../Headers)

foreach( f ${files} )
  set( src ${CMAKE_CURRENT_SOURCE_DIR}/${f} )
  set( dst ${output_dir}/${f} )
  add_custom_command(OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying clang's ${f}...")
endforeach( f )

add_custom_target(clang_headers ALL
  DEPENDS ${files})

install(FILES ${files}
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION Headers)
