// RUN: clang-cc -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

// CHECK: [[i64_i64_ty:%.*]] = type { i64, i64 }
// CHECK: [[i64_double_ty:%.*]] = type { i64, double }

// Basic base class test.
struct f0_s0 { unsigned a; };
struct f0_s1 : public f0_s0 { void *b; };
// CHECK: define void @_Z2f05f0_s1([[i64_i64_ty]])
void f0(f0_s1 a0) { }

// Check with two eight-bytes in base class.
struct f1_s0 { unsigned a; unsigned b; float c; };
struct f1_s1 : public f1_s0 { float d;};
// CHECK: define void @_Z2f15f1_s1([[i64_double_ty]])
void f1(f1_s1 a0) { }

// Check with two eight-bytes in base class and merge.
struct f2_s0 { unsigned a; unsigned b; float c; };
struct f2_s1 : public f2_s0 { char d;};
// CHECK: define void @_Z2f25f2_s1([[i64_i64_ty]])
void f2(f2_s1 a0) { }


