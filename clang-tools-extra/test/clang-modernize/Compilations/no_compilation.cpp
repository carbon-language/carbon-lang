// The following block tests:
//   - Neither -p nor -- was specified and a compilation database is detected
//     from source0 but the file isn't found the compilation database then
//     it's transformed using a fixed compilation database with c++11 support.
//     (-- -std=c++11).

// Create directory structure
// a1, a2  and a3 are specified paths for files in the compilation database but
// not a4.
// RUN: rm -rf %T/NoCompilation
// RUN: mkdir -p %T/NoCompilation
// RUN: mkdir -p %T/NoCompilation/a1
// RUN: mkdir -p %T/NoCompilation/a2
// RUN: mkdir -p %T/NoCompilation/a3
// RUN: mkdir -p %T/NoCompilation/a4

// This test uses of a compilation database
// RUN: sed -e 's#$(path)#%/T/NoCompilation#g' %S/Inputs/compile_commands.json > %T/NoCompilation/compile_commands.json

// RUN: cp %S/Inputs/cpp11.cpp %T/NoCompilation/a4
// RUN: clang-modernize -use-nullptr %T/NoCompilation/a4/cpp11.cpp
// RUN: diff -b %S/Inputs/cpp11_expected.cpp %T/NoCompilation/a4/cpp11.cpp
