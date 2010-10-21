// RUN: %llvmgcc -fno-exceptions %s -S -emit-llvm -o - | FileCheck %s
struct Foo
{
  int x;
  Foo ();
};
Foo *test(void)
{
  return new Foo();
  // There should be no references to any Unwinding routines under -fno-exceptions.
  // CHECK-NOT: Unwind
}
