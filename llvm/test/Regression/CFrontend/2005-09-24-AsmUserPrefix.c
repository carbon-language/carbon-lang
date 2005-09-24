// RUN: %llvmgcc %s -S -o - | gccas &&
// RUN: %llvmgcc %s -S -o - | gccas | llc &&
// RUN: %llvmgcc %s -S -o - | gccas | llc | not grep _foo2

void foo() __asm__("foo2");

void bar() {
  foo();
}
