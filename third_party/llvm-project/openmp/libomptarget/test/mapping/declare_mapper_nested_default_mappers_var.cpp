// RUN: %libomptarget-compilexx-run-and-check-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-x86_64-pc-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-nvptx64-nvidia-cuda

#include <cstdio>
#include <cstdlib>

typedef struct {
  int a;
  double *b;
} C1;
#pragma omp declare mapper(C1 s) map(to : s.a) map(from : s.b [0:2])

typedef struct {
  int a;
  double *b;
  C1 c;
} C;
#pragma omp declare mapper(C s) map(to : s.a, s.c) map(from : s.b [0:2])

typedef struct {
  int e;
  C f;
  int h;
} D;

int main() {
  constexpr int N = 10;
  D s;
  s.e = 111;
  s.f.a = 222;
  s.f.c.a = 777;
  double x[2];
  double x1[2];
  x[1] = 20;
  s.f.b = &x[0];
  s.f.c.b = &x1[0];
  s.h = N;

  printf("%d %d %d %4.5f %d\n", s.e, s.f.a, s.f.c.a, s.f.b[1],
         s.f.b == &x[0] ? 1 : 0);
  // CHECK: 111 222 777 20.00000 1

  __intptr_t p = reinterpret_cast<__intptr_t>(&x[0]);

#pragma omp target map(tofrom : s) firstprivate(p)
  {
    printf("%d %d %d\n", s.f.a, s.f.c.a,
           s.f.b == reinterpret_cast<void *>(p) ? 1 : 0);
    // CHECK: 222 777 0
    s.e = 333;
    s.f.a = 444;
    s.f.c.a = 555;
    s.f.b[1] = 40;
  }

  printf("%d %d %d %4.5f %d\n", s.e, s.f.a, s.f.c.a, s.f.b[1],
         s.f.b == &x[0] ? 1 : 0);
  // CHECK: 333 222 777 40.00000 1
}
