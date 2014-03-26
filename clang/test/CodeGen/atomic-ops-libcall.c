// RUN: %clang_cc1 < %s -triple armv5e-none-linux-gnueabi -emit-llvm -O1 | FileCheck %s

enum memory_order {
  memory_order_relaxed, memory_order_consume, memory_order_acquire,
  memory_order_release, memory_order_acq_rel, memory_order_seq_cst
};

int *test_c11_atomic_fetch_add_int_ptr(_Atomic(int *) *p) {
  // CHECK: test_c11_atomic_fetch_add_int_ptr
  // CHECK: {{%[^ ]*}} = tail call i32* @__atomic_fetch_add_4(i8* {{%[0-9]+}}, i32 12, i32 5)
  return __c11_atomic_fetch_add(p, 3, memory_order_seq_cst);
}

int *test_c11_atomic_fetch_sub_int_ptr(_Atomic(int *) *p) {
  // CHECK: test_c11_atomic_fetch_sub_int_ptr
  // CHECK: {{%[^ ]*}} = tail call i32* @__atomic_fetch_sub_4(i8* {{%[0-9]+}}, i32 20, i32 5)
  return __c11_atomic_fetch_sub(p, 5, memory_order_seq_cst);
}

int test_c11_atomic_fetch_add_int(_Atomic(int) *p) {
  // CHECK: test_c11_atomic_fetch_add_int
  // CHECK: {{%[^ ]*}} = tail call i32 bitcast (i32* (i8*, i32, i32)* @__atomic_fetch_add_4 to i32 (i8*, i32, i32)*)(i8* {{%[0-9]+}}, i32 3, i32 5)
  return __c11_atomic_fetch_add(p, 3, memory_order_seq_cst);
}

int test_c11_atomic_fetch_sub_int(_Atomic(int) *p) {
  // CHECK: test_c11_atomic_fetch_sub_int
  // CHECK: {{%[^ ]*}} = tail call i32 bitcast (i32* (i8*, i32, i32)* @__atomic_fetch_sub_4 to i32 (i8*, i32, i32)*)(i8* {{%[0-9]+}}, i32 5, i32 5)
  return __c11_atomic_fetch_sub(p, 5, memory_order_seq_cst);
}

int *fp2a(int **p) {
  // CHECK: @fp2a
  // CHECK: {{%[^ ]*}} = tail call i32* @__atomic_fetch_sub_4(i8* {{%[0-9]+}}, i32 4, i32 0)
  // Note, the GNU builtins do not multiply by sizeof(T)!
  return __atomic_fetch_sub(p, 4, memory_order_relaxed);
}
