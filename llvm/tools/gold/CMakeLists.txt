set(LLVM_BINUTILS_INCDIR "" CACHE PATH
  "PATH to binutils/include containing plugin-api.h for gold plugin.")

set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/gold.exports)

if( NOT LLVM_BINUTILS_INCDIR )
  # Nothing to say.
elseif( NOT EXISTS "${LLVM_BINUTILS_INCDIR}/plugin-api.h" )
  message(STATUS "plugin-api.h not found. gold plugin excluded from the build.")
else()
  include_directories( ${LLVM_BINUTILS_INCDIR} )

  # Because off_t is used in the public API, the largefile parts are required for
  # ABI compatibility.
  add_definitions( -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 )

  set(LLVM_LINK_COMPONENTS support)

  add_llvm_loadable_module(LLVMgold
    gold-plugin.cpp
    )
endif()

target_link_libraries(LLVMgold LTO)
