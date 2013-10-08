set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  ipo scalaropts linker bitreader bitwriter lto mcdisassembler vectorize)

add_definitions( -DLLVM_VERSION_INFO=\"${PACKAGE_VERSION}\" )

set(SOURCES
  LTODisassembler.cpp
  lto.cpp
  )

if( NOT CYGWIN AND LLVM_ENABLE_PIC )
  if ( WIN32 )
    # Create .def file containing a list of exports preceeded by
    # 'EXPORTS'.  The file "lto.exports" already contains the list, so we
    # massage it into the correct format here to create "lto.exports.def".
    set(LTO_EXPORTS_DEF ${CMAKE_CURRENT_BINARY_DIR}/lto.exports.def)
    set(LTO_EXPORTS_DEF_TEMP ${LTO_EXPORTS_DEF}.txt)
    file(READ "lto.exports" exports_list)
    file(WRITE ${LTO_EXPORTS_DEF_TEMP} "LIBRARY LTO\n")
    file(APPEND ${LTO_EXPORTS_DEF_TEMP} "EXPORTS\n")
    file(APPEND ${LTO_EXPORTS_DEF_TEMP} ${exports_list})

    # Copy the file only if it has changed.
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${LTO_EXPORTS_DEF_TEMP} ${LTO_EXPORTS_DEF})

    set(SHARED_LIB_SOURCES ${SOURCES} ${LTO_EXPORTS_DEF})
  else()
    set(SHARED_LIB_SOURCES ${SOURCES})
  endif()

  set(bsl ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS ON)
  add_llvm_library(LTO ${SHARED_LIB_SOURCES})
  set_property(TARGET LTO PROPERTY OUTPUT_NAME "LTO")
  set(BUILD_SHARED_LIBS ${bsl})
  set(LTO_STATIC_TARGET_NAME LTO_static)
else()
  set(LTO_STATIC_TARGET_NAME LTO)
endif()

if( NOT BUILD_SHARED_LIBS )
  add_llvm_library(${LTO_STATIC_TARGET_NAME} ${SOURCES})
  set_property(TARGET ${LTO_STATIC_TARGET_NAME} PROPERTY OUTPUT_NAME "LTO")
endif()

if( NOT CYGWIN )
  install(FILES ${LLVM_MAIN_INCLUDE_DIR}/llvm-c/lto.h
          DESTINATION include/llvm-c)
endif()
