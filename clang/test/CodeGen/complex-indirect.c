// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin10 | FileCheck %s

// Make sure this doesn't crash, and that we don't generate a byval alloca
// with insufficient alignment.

void a(int,int,int,int,int,int,__complex__ char);
void b(__complex__ char *y) { a(0,0,0,0,0,0,*y); }
// CHECK: define void @b
// CHECK: alloca { i8, i8 }*, align 8
// CHECK: call void @a(i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, { i8, i8 }* byval align 8
