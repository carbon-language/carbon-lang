// The following block tests:
//   - A compilation database is detected from source0.

// Create directory structure
// a1, a2  and a3 are specified paths for files in the compilation database.
// RUN: rm -rf %T/DetectFromSource
// RUN: mkdir -p %T/DetectFromSource
// RUN: mkdir -p %T/DetectFromSource/a1
// RUN: mkdir -p %T/DetectFromSource/a2
// RUN: mkdir -p %T/DetectFromSource/a3

// This test uses a compilation database
// RUN: sed -e 's#$(path)#%/T/DetectFromSource#g' %S/Inputs/compile_commands.json > %T/DetectFromSource/compile_commands.json

// Check that a compilation database can be auto-detected from source0
// RUN: cp %S/Inputs/compilations.cpp %T/DetectFromSource/a1
// RUN: cp %S/Inputs/compilations.cpp %T/DetectFromSource/a2
// RUN: cp %S/Inputs/compilations.cpp %T/DetectFromSource/a3

// RUN: clang-modernize -use-nullptr %T/DetectFromSource/a1/compilations.cpp %T/DetectFromSource/a3/compilations.cpp
// RUN: diff -b %S/Inputs/compilations_expected.cpp %T/DetectFromSource/a1/compilations.cpp
// RUN: not diff -b %S/Inputs/compilations_expected.cpp %T/DetectFromSource/a2/compilations.cpp
// RUN: diff -b %S/Inputs/compilations_expected.cpp %T/DetectFromSource/a3/compilations.cpp
