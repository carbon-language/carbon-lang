// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc-unknown-unknown -emit-llvm %s -o - | FileCheck %s

void test_eh_return_data_regno()
{
  volatile int res;
  res = __builtin_eh_return_data_regno(0);  // CHECK: store volatile i32 3
  res = __builtin_eh_return_data_regno(1);  // CHECK: store volatile i32 4
}

// CHECK-LABEL: define{{.*}} i64 @test_builtin_ppc_get_timebase
long long test_builtin_ppc_get_timebase() {
  // CHECK: call i64 @llvm.readcyclecounter()
  return __builtin_ppc_get_timebase();
}

void test_builtin_ppc_setrnd() {
  volatile double res;
  volatile int x = 100;
  
  // CHECK: call double @llvm.ppc.setrnd(i32 2)
  res = __builtin_setrnd(2);

  // CHECK: call double @llvm.ppc.setrnd(i32 100)
  res = __builtin_setrnd(100);

  // CHECK: call double @llvm.ppc.setrnd(i32 %2)
  res = __builtin_setrnd(x);
}

void test_builtin_ppc_flm() {
  volatile double res;
  // CHECK: call double @llvm.ppc.readflm()
  res = __builtin_readflm();

  // CHECK: call double @llvm.ppc.setflm(double %1)
  res = __builtin_setflm(res);
}

double test_builtin_unpack_ldbl(long double x) {
  // CHECK: call double @llvm.ppc.unpack.longdouble(ppc_fp128 %0, i32 1)
  return __builtin_unpack_longdouble(x, 1);
}

long double test_builtin_pack_ldbl(double x, double y) {
  // CHECK: call ppc_fp128 @llvm.ppc.pack.longdouble(double %0, double %1)
  return __builtin_pack_longdouble(x, y);
}
