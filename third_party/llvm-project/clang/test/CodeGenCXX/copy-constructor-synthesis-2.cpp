// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -emit-llvm -o - %s | FileCheck %s

union PR23373 {
  PR23373(PR23373&) = default;
  PR23373 &operator=(PR23373&) = default;
  int n;
  float f;
};
extern PR23373 pr23373_a;

PR23373 pr23373_b(pr23373_a);
// CHECK-LABEL: define {{.*}} @__cxx_global_var_init(
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i32|i64}}({{.*}}align 4{{.*}}@pr23373_b{{.*}}, {{.*}}align 4{{.*}} @pr23373_a{{.*}}, [[W:i32|i64]] 4, i1 false)

PR23373 pr23373_f() { return pr23373_a; }
// CHECK-LABEL: define {{.*}} @_Z9pr23373_fv(
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.[[W]]({{.*}}align 4{{.*}}align 4{{.*}}, [[W]] 4, i1 false)

void pr23373_g(PR23373 &a, PR23373 &b) { a = b; }
// CHECK-LABEL: define {{.*}} @_Z9pr23373_g
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.[[W]]({{.*}}align 4{{.*}}align 4{{.*}}, [[W]] 4, i1 false)

struct A { virtual void a(); };
A x(A& y) { return y; }

// CHECK: define linkonce_odr {{.*}} @_ZN1AC1ERKS_(%struct.A* {{.*}}%this, %struct.A* nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0) unnamed_addr
// CHECK: store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV1A, i32 0, inrange i32 0, i32 2) to i32 (...)**)
