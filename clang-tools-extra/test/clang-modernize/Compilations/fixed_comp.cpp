// The following block tests that files are transformed when -- is specified.

// Create directory structure
// a1, a2  and a3 are specified paths for files in the compilation database.
// RUN: rm -rf %T/FixedComp
// RUN: mkdir -p %T/FixedComp
// RUN: mkdir -p %T/FixedComp/a1
// RUN: mkdir -p %T/FixedComp/a2
// RUN: mkdir -p %T/FixedComp/a3

// RUN: cp %S/Inputs/compilations.cpp %T/FixedComp/a1
// RUN: cp %S/Inputs/compilations.cpp %T/FixedComp/a2
// RUN: cp %S/Inputs/compilations.cpp %T/FixedComp/a3

// RUN: clang-modernize -use-nullptr %T/FixedComp/a1/compilations.cpp %T/FixedComp/a3/compilations.cpp --
// RUN: diff -b %S/Inputs/compilations_expected.cpp %T/FixedComp/a1/compilations.cpp
// RUN: not diff -b %S/Inputs/compilations_expected.cpp %T/FixedComp/a2/compilations.cpp
// RUN: diff -b %S/Inputs/compilations_expected.cpp %T/FixedComp/a3/compilations.cpp
