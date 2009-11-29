// RUN: clang-cc -emit-llvm -o %t %s

typedef short T[4];
struct s {
  T f0;
};

void foo(struct s *x) {
  bar((long) x->f0);
}
