// RUN: %clang_cc1 -S -emit-llvm -debug-info-kind=limited  %s -o - | FileCheck %s
// CHECK: !DIGlobalVariable({{.*}}
// CHECK-NOT: expr:

static const __uint128_t ro = 18446744073709551615;

void bar(__uint128_t);
void foo() { bar(ro); }
