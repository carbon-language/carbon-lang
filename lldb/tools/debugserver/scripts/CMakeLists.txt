set(LLVM_NO_RTTI 1)

set(dnbconfig_builder ${CMAKE_CURRENT_SOURCE_DIR}/dbgnub-config.pl)
set(dnbconfig_header ${LLDB_BINARY_DIR}/tools/debugserver/DNBConfig.h)
set_source_files_properties(${dnbconfig_header} PROPERTIES GENERATED 1)

add_custom_command(OUTPUT ${dnbconfig_header}
  COMMAND SCRIPT_OUTPUT_FILE_0=${dnbconfig_header} perl ${dnbconfig_builder}
  DEPENDS ${dnbconfig_builder}
  )
add_custom_target(generate_dnbconfig
  DEPENDS ${dnbconfig_header}
  )

