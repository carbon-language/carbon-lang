// Don't attempt slash switches on msys bash.
// REQUIRES: shell-preserves-root

// RUN: %clang_cl /c /W0 %s -### 2>&1 | FileCheck -check-prefix=W0 %s
// W0-DAG: -c
// W0-DAG: -w

// RUN: %clang_cl /c /W1 %s -### 2>&1 | FileCheck -check-prefix=W1 %s
// W1-DAG: -c
// W1-DAG: -Wall
