// RUN: %clang_cc1 %s -emit-llvm -O0 -o - | FileCheck %s
// PR 5406

// XFAIL: *
// XTARGET: arm

typedef struct { char x[3]; } A0;
void foo (int i, ...);


// CHECK: call void (i32, ...)* @foo(i32 1, i32 {{.*}}) nounwind
int main (void)
{
  A0 a3;
  a3.x[0] = 0;
  a3.x[0] = 0;
  a3.x[2] = 26;
  foo (1,  a3 );
  return 0;
}
