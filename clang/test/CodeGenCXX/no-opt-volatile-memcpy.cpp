// RUN: %clang_cc1 -O -triple=x86_64-apple-darwin  -emit-llvm -o - %s | FileCheck %s
// rdar://11861085

struct s {
  char filler [128];
  volatile int x;
};

struct s gs;

void foo (void) {
  struct s ls;
  ls = ls;
  gs = gs;
  ls = gs;
}
// CHECK: call void @llvm.memcpy
// CHECK: call void @llvm.memcpy
// CHECK: call void @llvm.memcpy

struct s1 {
  struct s y;
};

struct s1 s;

void fee (void) {
  s = s;
  s.y = gs;
}
// CHECK: call void @llvm.memcpy
// CHECK: call void @llvm.memcpy


struct d : s1 {
};

d gd;

void gorf(void) {
  gd = gd;
}
// CHECK: call void @llvm.memcpy

