// RUN: %clang_cc1 -triple i386-unknown-unknown %s -emit-llvm -o - | FileCheck %s

#define FASTCALL __attribute__((regparm(2)))

typedef struct {
  int aaa;
  double bbbb;
  int ccc[200];
} foo;

static void FASTCALL
reduced(char b, double c, foo* d, double e, int f) {
}

int
main(void) {
  // CHECK: call void @reduced(i8 signext inreg 0, {{.*}} %struct.anon* inreg null
  reduced(0, 0.0, 0, 0.0, 0);
}
