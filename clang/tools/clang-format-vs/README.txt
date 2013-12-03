This directory contains a VSPackage project to generate a Visual Studio extension
for clang-format.

Build prerequisites are:
- Visual Studio 2012 Professional
- Visual Studio 2010 Professional
- Visual Studio 2010 SDK.

clang-format.exe must be copied into the ClangFormat/ directory before building.
It will be bundled into the .vsix file.

The extension can be built manually from ClangFormat.sln (e.g. by opening it in
Visual Studio), or with cmake by setting the BUILD_CLANG_FORMAT_VS_PLUGIN flag.
