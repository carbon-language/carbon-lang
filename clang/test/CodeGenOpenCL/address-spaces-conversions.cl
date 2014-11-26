// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -O0 -ffake-address-space-map -cl-std=CL2.0 -emit-llvm -o - | FileCheck %s

// test that we generate address space casts everywhere we need conversions of
// pointers to different address spaces

void test(global int *arg_glob, generic int *arg_gen) {
  int var_priv;
  arg_gen = arg_glob; // implicit cast global -> generic
  // CHECK: %{{[0-9]+}} = addrspacecast i32 addrspace(1)* %{{[0-9]+}} to i32 addrspace(4)*
  arg_gen = &var_priv; // implicit cast with obtaining adr, private -> generic
  // CHECK: %{{[0-9]+}} = addrspacecast i32* %var_priv to i32 addrspace(4)*
  arg_glob = (global int *)arg_gen; // explicit cast
  // CHECK: %{{[0-9]+}} = addrspacecast i32 addrspace(4)* %{{[0-9]+}} to i32 addrspace(1)*
  global int *var_glob =
      (global int *)arg_glob; // explicit cast in the same address space
  // CHECK-NOT: %{{[0-9]+}} = addrspacecast i32 addrspace(1)* %{{[0-9]+}} to i32 addrspace(1)*
  var_priv = arg_gen - arg_glob; // arithmetic operation
  // CHECK: %{{.*}} = ptrtoint i32 addrspace(4)* %{{.*}} to i64
  // CHECK: %{{.*}} = ptrtoint i32 addrspace(1)* %{{.*}} to i64
  var_priv = arg_gen > arg_glob; // comparison
  // CHECK: %{{[0-9]+}} = addrspacecast i32 addrspace(1)* %{{[0-9]+}} to i32 addrspace(4)*
}
