#include <stdio.h>
//extern int printf(const char *, ...);

int CN = 0;
int DN = 0;

struct foo {
  int Num;
  foo(int num) : Num(num) {
    printf("Foo ctor %d %d\n", Num, CN++);
  }
  ~foo() {
    printf("Foo dtor %d %d\n", Num, DN++);
  }
} Constructor1(7);     // Global with ctor to be called before main
foo Constructor2(12);

struct bar {
  ~bar() {
    printf("bar dtor\n");
  }
} Destructor1;     // Global with dtor

int main() {
  printf("main\n");
  return 0;
}
