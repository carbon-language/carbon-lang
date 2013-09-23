// Test that only specified sources are transformed when -p and -include are
// specified along with sources.

// Create directory structure
// a1, a2  and a3 are specified paths for files in the compilation database.
// RUN: rm -rf %T/CompilationIncSources
// RUN: mkdir -p %T/CompilationIncSources
// RUN: mkdir -p %T/CompilationIncSources/a1
// RUN: mkdir -p %T/CompilationIncSources/a2
// RUN: mkdir -p %T/CompilationIncSources/a3

// This test uses a compilation database
// RUN: sed -e 's#$(path)#%/T/CompilationIncSources#g' %S/Inputs/compile_commands.json > %T/CompilationIncSources/compile_commands.json

// RUN: cp %S/Inputs/compilations.cpp %T/CompilationIncSources/a1
// RUN: cp %S/Inputs/compilations.cpp %T/CompilationIncSources/a2
// RUN: cp %S/Inputs/compilations.cpp %T/CompilationIncSources/a3

// RUN: clang-modernize -use-nullptr -p=%T/CompilationIncSources -include=%T/CompilationIncSources %T/CompilationIncSources/a2/compilations.cpp
// RUN: not diff -b %S/Inputs/compilations_expected.cpp %T/CompilationIncSources/a1/compilations.cpp
// RUN: diff -b %S/Inputs/compilations_expected.cpp %T/CompilationIncSources/a2/compilations.cpp
// RUN: not diff -b %S/Inputs/compilations_expected.cpp %T/CompilationIncSources/a3/compilations.cpp
