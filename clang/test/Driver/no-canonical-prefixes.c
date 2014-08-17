// REQUIRES: shell
// RUN: mkdir -p %t
// RUN: cd %t
// RUN: ln -sf %clang test-clang
// RUN: ./test-clang -v -S %s 2>&1 | FileCheck %s
// RUN: ./test-clang -v -S %s -no-canonical-prefixes 2>&1 | FileCheck --check-prefix=NCP %s


// CHECK: /clang{{.*}}" -cc1
// NCP: test-clang" -cc1
