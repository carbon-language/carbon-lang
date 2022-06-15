// RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -DDOUBLE -O0 -o - -triple=amdgcn-amd-amdhsa \
// RUN:   | opt -instnamer -S | FileCheck -check-prefixes=FLOAT,DOUBLE %s

// RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -DDOUBLE -O0 -o - -triple=aarch64-linux-gnu \
// RUN:   | opt -instnamer -S | FileCheck -check-prefixes=FLOAT,DOUBLE %s

// RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -O0 -o - -triple=armv8-apple-ios7.0 \
// RUN:   | opt -instnamer -S | FileCheck -check-prefixes=FLOAT %s

// RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -DDOUBLE -O0 -o - -triple=hexagon \
// RUN:   | opt -instnamer -S | FileCheck -check-prefixes=FLOAT,DOUBLE %s

// RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -DDOUBLE -O0 -o - -triple=mips64-mti-linux-gnu \
// RUN:   | opt -instnamer -S | FileCheck -check-prefixes=FLOAT,DOUBLE %s

// RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -O0 -o - -triple=i686-linux-gnu \
// RUN:   | opt -instnamer -S | FileCheck -check-prefixes=FLOAT %s

// RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -DDOUBLE -O0 -o - -triple=x86_64-linux-gnu \
// RUN:   | opt -instnamer -S | FileCheck -check-prefixes=FLOAT,DOUBLE %s

typedef enum memory_order {
  memory_order_relaxed = __ATOMIC_RELAXED,
  memory_order_acquire = __ATOMIC_ACQUIRE,
  memory_order_release = __ATOMIC_RELEASE,
  memory_order_acq_rel = __ATOMIC_ACQ_REL,
  memory_order_seq_cst = __ATOMIC_SEQ_CST
} memory_order;

void test(float *f, float ff, double *d, double dd) {
  // FLOAT: atomicrmw fadd float* {{.*}} monotonic
  __atomic_fetch_add(f, ff, memory_order_relaxed);

  // FLOAT: atomicrmw fsub float* {{.*}} monotonic
  __atomic_fetch_sub(f, ff, memory_order_relaxed);

#ifdef DOUBLE
  // DOUBLE: atomicrmw fadd double* {{.*}} monotonic
  __atomic_fetch_add(d, dd, memory_order_relaxed);

  // DOUBLE: atomicrmw fsub double* {{.*}} monotonic
  __atomic_fetch_sub(d, dd, memory_order_relaxed);
#endif
}
