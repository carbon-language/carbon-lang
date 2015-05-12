option(BUILD_CLANG_FORMAT_VS_PLUGIN "Build clang-format VS plugin" OFF)
if (BUILD_CLANG_FORMAT_VS_PLUGIN)
  add_custom_target(clang_format_exe_for_vsix
      ${CMAKE_COMMAND} -E copy_if_different
      "${LLVM_TOOLS_BINARY_DIR}/clang-format.exe"
      "${CMAKE_CURRENT_SOURCE_DIR}/ClangFormat/clang-format.exe"
      DEPENDS clang-format)

  add_custom_target(clang_format_license
      ${CMAKE_COMMAND} -E copy_if_different
      "${CLANG_SOURCE_DIR}/LICENSE.TXT"
      "${CMAKE_CURRENT_SOURCE_DIR}/ClangFormat/license.txt")

  if (NOT CLANG_FORMAT_VS_VERSION)
    set(CLANG_FORMAT_VS_VERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
  endif()

  configure_file("source.extension.vsixmanifest.in"
      "${CMAKE_CURRENT_SOURCE_DIR}/ClangFormat/source.extension.vsixmanifest")

  add_custom_target(clang_format_vsix ALL
      devenv "${CMAKE_CURRENT_SOURCE_DIR}/ClangFormat.sln" /Build Release
      DEPENDS clang_format_exe_for_vsix "${CMAKE_CURRENT_SOURCE_DIR}/ClangFormat/source.extension.vsixmanifest"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_CURRENT_SOURCE_DIR}/ClangFormat/bin/Release/ClangFormat.vsix"
      "${LLVM_TOOLS_BINARY_DIR}/ClangFormat.vsix"
      DEPENDS clang_format_exe_for_vsix clang_format_license)
endif()
