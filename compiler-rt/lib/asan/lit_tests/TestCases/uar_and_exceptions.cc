// Test that use-after-return works with exceptions.
// RUN: %clangxx_asan -fsanitize=use-after-return -O0 %s -o %t && %t

#include <stdio.h>

volatile char *g;

void Func(int depth) {
  char frame[100];
  g = &frame[0];
  if (depth)
    Func(depth - 1);
  else
    throw 1;
}

int main(int argc, char **argv) {
  for (int i = 0; i < 4000; i++) {
    try {
      Func(argc * 100);
    } catch(...) {
    }
    if ((i % 1000) == 0)
      fprintf(stderr, "done [%d]\n", i);
  }
  return 0;
}
