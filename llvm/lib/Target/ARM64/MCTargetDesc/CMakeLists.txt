add_llvm_library(LLVMARM64Desc
  ARM64AsmBackend.cpp
  ARM64BaseInfo.cpp
  ARM64ELFObjectWriter.cpp
  ARM64ELFStreamer.cpp
  ARM64MCAsmInfo.cpp
  ARM64MCCodeEmitter.cpp
  ARM64MCExpr.cpp
  ARM64MCTargetDesc.cpp
  ARM64MachObjectWriter.cpp
)
add_dependencies(LLVMARM64Desc ARM64CommonTableGen)

# Hack: we need to include 'main' target directory to grab private headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/..)
