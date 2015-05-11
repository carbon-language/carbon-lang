// RUN: %clang -no-canonical-prefixes --target=unknown-unknown-unknown -c %s \
// RUN:   -o %t.o -### 2>&1 | FileCheck %s
//
// Ensure we get a crazy triple here as we asked for one.
// CHECK: Target: unknown-unknown-unknown
//
// Also check that the legacy spelling works.
// RUN: %clang -no-canonical-prefixes -target unknown-unknown-unknown -c %s \
// RUN:   -o %t.o -### 2>&1 | FileCheck %s
