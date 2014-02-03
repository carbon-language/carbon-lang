// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-tidy %t.cpp -- -fan-unknown-option > %t2.cpp
// RUN: FileCheck -input-file=%t2.cpp %s

// CHECK: warning: unknown argument: '-fan-unknown-option'
