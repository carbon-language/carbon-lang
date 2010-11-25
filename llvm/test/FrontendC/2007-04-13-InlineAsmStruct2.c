// RUN: %llvmgcc %s -S -o - | grep {call void asm}

struct V { short X, Y; };
int bar() {
  struct V bar;
  __asm__ volatile("foo %0\n" :: "r"(bar));
  return bar.X;
}

