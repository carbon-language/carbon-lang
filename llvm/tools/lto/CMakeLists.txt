set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  LTO
  MC
  Support
  )

add_definitions( -DLLVM_VERSION_INFO=\"${PACKAGE_VERSION}\" )

set(SOURCES
  LTODisassembler.cpp
  lto.cpp
  )

set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/lto.exports)

if( NOT CYGWIN AND LLVM_ENABLE_PIC )
  set(SHARED_LIB_SOURCES ${SOURCES})

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
