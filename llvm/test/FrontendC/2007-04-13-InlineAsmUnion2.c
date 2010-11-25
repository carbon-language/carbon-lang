// RUN: %llvmgcc %s -S -o - | grep {call void asm}

union U { int x; char* p; };
void foo() {
  union U bar;
  __asm__ volatile("foo %0\n" :: "r"(bar));
}
