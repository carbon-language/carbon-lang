// RUN: %clang_cc1 -triple i386-unknown-unknown %s -emit-llvm -o - | FileCheck %s


// CHECK: _Z3fooRi(i32* inreg
void __attribute__ ((regparm (1)))  foo(int &a) {
}
