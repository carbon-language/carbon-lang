set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  ipo scalaropts linker bitreader bitwriter)

add_definitions( -DLLVM_VERSION_INFO=\"${PACKAGE_VERSION}\" )

set(SOURCES
  LTOCodeGenerator.cpp
  lto.cpp
  LTOModule.cpp
  )

add_llvm_library(LTO ${SOURCES})

if( NOT WIN32 AND LLVM_ENABLE_PIC )
  set(BUILD_SHARED_LIBS ON)
  add_llvm_library(LTO_dll ${SOURCES})
  set_property(TARGET LTO_dll PROPERTY OUTPUT_NAME "LTO")
endif()
