// RUN: clang -g -emit-llvm -o %t %s

struct s0;
struct s0 { struct s0 *p; } g0;

struct s0 *f0(struct s0 *a0) {
  return a0->p;
}
  
  
