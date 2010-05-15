// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

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

// PR5831
struct s3_0 {};
struct s3_1 { struct s3_0 a; long b; };
void f3(struct s3_1 x) {}

// CHECK: define i64 @_Z4f4_0M2s4i(i64)
// CHECK: define [[i64_i64_ty]] @_Z4f4_1M2s4FivE([[i64_i64_ty]])
struct s4 {};
typedef int s4::* s4_mdp;
typedef int (s4::*s4_mfp)();
s4_mdp f4_0(s4_mdp a) { return a; }
s4_mfp f4_1(s4_mfp a) { return a; }
