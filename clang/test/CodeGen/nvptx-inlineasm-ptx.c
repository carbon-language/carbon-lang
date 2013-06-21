// RUN: %clang_cc1 -triple nvptx-unknown-unknown -O3 -S -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -O3 -S -o - %s | FileCheck %s

void constraints() {
  char           c;
  unsigned char  uc;
  short          s;
  unsigned short us;
  int            i;
  unsigned int   ui;
  long           l;
  unsigned long  ul;
  float          f;
  double         d;

  // CHECK: mov.b8 %rc{{[0-9]+}}, %rc{{[0-9]+}}
  asm volatile ("mov.b8 %0, %1;" : "=c"(c) : "c"(c));
  // CHECK: mov.b8 %rc{{[0-9]+}}, %rc{{[0-9]+}}
  asm volatile ("mov.b8 %0, %1;" : "=c"(uc) : "c"(uc));

  // CHECK: mov.b16 %rs{{[0-9]+}}, %rs{{[0-9]+}}
  asm volatile ("mov.b16 %0, %1;" : "=h"(s) : "h"(s));
  // CHECK: mov.b16 %rs{{[0-9]+}}, %rs{{[0-9]+}}
  asm volatile ("mov.b16 %0, %1;" : "=h"(us) : "h"(us));

  // CHECK: mov.b32 %r{{[0-9]+}}, %r{{[0-9]+}}
  asm volatile ("mov.b32 %0, %1;" : "=r"(i) : "r"(i));
  // CHECK: mov.b32 %r{{[0-9]+}}, %r{{[0-9]+}}
  asm volatile ("mov.b32 %0, %1;" : "=r"(ui) : "r"(ui));

  // CHECK: mov.b64 %rl{{[0-9]+}}, %rl{{[0-9]+}}
  asm volatile ("mov.b64 %0, %1;" : "=l"(l) : "l"(l));
  // CHECK: mov.b64 %rl{{[0-9]+}}, %rl{{[0-9]+}}
  asm volatile ("mov.b64 %0, %1;" : "=l"(ul) : "l"(ul));

  // CHECK: mov.b32 %f{{[0-9]+}}, %f{{[0-9]+}}
  asm volatile ("mov.b32 %0, %1;" : "=f"(f) : "f"(f));
  // CHECK: mov.b64 %fl{{[0-9]+}}, %fl{{[0-9]+}}
  asm volatile ("mov.b64 %0, %1;" : "=d"(d) : "d"(d));
}
