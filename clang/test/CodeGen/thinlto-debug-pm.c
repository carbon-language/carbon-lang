// Test to ensure -fdebug-pass-manager works when invoking the
// ThinLTO backend path with the new PM.
// RUN: %clang -O2 %s -flto=thin -c -o %t.o
// RUN: llvm-lto -thinlto -o %t %t.o
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O2 -o %t2.o -x ir %t.o -fthinlto-index=%t.thinlto.bc -fdebug-pass-manager -fexperimental-new-pass-manager 2>&1 | FileCheck %s
// CHECK: Running pass:

void foo() {
}
