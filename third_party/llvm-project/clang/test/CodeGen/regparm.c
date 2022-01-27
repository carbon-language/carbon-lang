// RUN: %clang_cc1 -triple i386-unknown-unknown %s -emit-llvm -o - | FileCheck %s

#define FASTCALL __attribute__((regparm(2)))

typedef struct {
  int aaa;
  double bbbb;
  int ccc[200];
} foo;

typedef void (*FType)(int, int)      __attribute ((regparm (3), stdcall));
FType bar;

extern void FASTCALL reduced(char b, double c, foo* d, double e, int f);

// PR7025
void FASTCALL f1(int i, int j, int k);
// CHECK-LABEL: define{{.*}} void @f1(i32 inreg %i, i32 inreg %j, i32 %k)
void f1(int i, int j, int k) { }

int
main(void) {
  // CHECK: call void @reduced(i8 inreg signext 0, {{.*}} %struct.foo* inreg null
  reduced(0, 0.0, 0, 0.0, 0);
  // CHECK: call x86_stdcallcc void {{.*}}(i32 inreg 1, i32 inreg 2)
  bar(1,2);
}
