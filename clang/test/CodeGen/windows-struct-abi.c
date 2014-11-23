// RUN: %clang_cc1 -triple i686-windows-itanium -emit-llvm -o - %s | FileCheck %s

struct f1 {
  float f;
};

struct f1 return_f1(void) { while (1); }

// CHECK: define void @return_f1(%struct.f1* noalias sret %agg.result)

void receive_f1(struct f1 a0) { }

// CHECK: define void @receive_f1(%struct.f1* byval align 4 %a0)

struct f2 {
  float f;
  float g;
};

struct f2 return_f2(void) { while (1); }

// CHECK: define void @return_f2(%struct.f2* noalias sret %agg.result)

void receive_f2(struct f2 a0) { }

// CHECK: define void @receive_f2(%struct.f2* byval align 4 %a0)

