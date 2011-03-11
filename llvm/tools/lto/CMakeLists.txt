set(LLVM_LINK_COMPONENTS ipo scalaropts linker bitreader bitwriter)

add_definitions( -DLLVM_VERSION_INFO=\"${PACKAGE_VERSION}\" )

# TODO: build a static library too.
set(BUILD_SHARED_LIBS ON)

add_llvm_library(LTO
  LTOCodeGenerator.cpp
  lto.cpp
  LTOModule.cpp
  )

