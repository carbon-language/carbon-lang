// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck %s

struct A { virtual void a(); };
A x(A& y) { return y; }

// CHECK: define linkonce_odr {{.*}} @_ZN1AC1ERKS_(%struct.A* {{.*}}%this, %struct.A* dereferenceable({{[0-9]+}})) unnamed_addr
// CHECK: store i32 (...)** bitcast (i8** getelementptr inbounds ([3 x i8*]* @_ZTV1A, i64 0, i64 2) to i32 (...)**)
