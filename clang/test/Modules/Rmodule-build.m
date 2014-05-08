// REQUIRES: shell
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo '// A' > %t/A.h
// RUN: echo '// B' > %t/B.h
// RUN: echo 'module A { header "A.h" }' > %t/module.modulemap
// RUN: echo 'module B { header "B.h" }' >> %t/module.modulemap

// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fsyntax-only %s -verify \
// RUN:            -I %t -Wmodule-build

@import A; // expected-remark{{building module 'A' as}}
@import B; // expected-remark{{building module 'B' as}}
@import A; // no diagnostic
@import B; // no diagnostic

// RUN: echo ' ' >> %t/B.h
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fsyntax-only %s -I %t \
// RUN:            -Wmodule-build 2>&1 | FileCheck %s

// CHECK-NOT: building module 'A'
// CHECK: building module 'B'
