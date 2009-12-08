// RUN: clang-cc -emit-llvm -o - %s | FileCheck %s

struct A { virtual void a(); };
A x(A& y) { return y; }

// CHECK: define linkonce_odr void @_ZN1AC1ERKS_(
// CHECK: store i8** getelementptr inbounds ([3 x i8*]* @_ZTV1A, i64 0, i64 2)
