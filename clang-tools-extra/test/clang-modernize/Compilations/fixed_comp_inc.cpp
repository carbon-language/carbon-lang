// The following block tests:
//   - A fixed compilation database is provided and -exclude was also used.

// Create directory structure
// a1, a2  and a3 are specified paths for files in the compilation database.
// RUN: rm -rf %T/FixedCompInc
// RUN: mkdir -p %T/FixedCompInc
// RUN: mkdir -p %T/FixedCompInc/a1
// RUN: mkdir -p %T/FixedCompInc/a2
// RUN: mkdir -p %T/FixedCompInc/a3

// Check that only files not explicitly excluded are transformed.
// RUN: cp %S/Inputs/compilations.cpp %T/FixedCompInc/a1
// RUN: cp %S/Inputs/compilations.cpp %T/FixedCompInc/a2
// RUN: cp %S/Inputs/compilations.cpp %T/FixedCompInc/a3

// RUN: clang-modernize -use-nullptr %T/FixedCompInc/a1/compilations.cpp %T/FixedCompInc/a2/compilations.cpp %T/FixedCompInc/a3/compilations.cpp -exclude=%T/FixedCompInc/a2 --
// RUN: diff -b %S/Inputs/compilations_expected.cpp %T/FixedCompInc/a1/compilations.cpp
// RUN: not diff -b %S/Inputs/compilations_expected.cpp %T/FixedCompInc/a2/compilations.cpp
// RUN: diff -b %S/Inputs/compilations_expected.cpp %T/FixedCompInc/a3/compilations.cpp
