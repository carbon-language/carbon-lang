// RUN: %llvmgcc %s -S -emit-llvm -O0 -o - | grep sret | wc -l | grep 5

struct abc {
 int a;
 int b;
 int c;
};
 
struct abc foo1(void);
struct abc foo2();

void bar() {
  struct abc dummy1 = foo1();
  struct abc dummy2 = foo2();
}
