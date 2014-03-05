// RUN: clang-tidy %s -- -fan-unknown-option | FileCheck -check-prefix=CHECK1 %s
// RUN: clang-tidy %s.nonexistent.cpp -- | FileCheck -check-prefix=CHECK2 %s

// CHECK1: warning: unknown argument: '-fan-unknown-option'
// CHECK2: warning: error reading '{{.*}}.nonexistent.cpp'
