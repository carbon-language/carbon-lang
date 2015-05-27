#include "instrprof-dynamic-header.h"
int X = 0;
void foo(int K) { if (K) {} }
int main(int argc, char *argv[]) {
  foo(5);
  X++;
  bar<void>();
  a();
  b();
  return 0;
}
