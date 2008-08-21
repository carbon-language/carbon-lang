// RUN: clang %s -emit-llvm -o %t
struct test;

typedef void (*my_func) (struct test *);
my_func handler;

struct test {
  char a;
};

char f(struct test *t) {
  return t->a;
}
