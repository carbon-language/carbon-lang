// RUN: clang-cc -triple i386-unknown-unknown --emit-llvm-bc -o - %s | opt --std-compile-opts | llvm-dis > %t
// RUN: grep "ret i32" %t | count 1
// RUN: grep "ret i32 1" %t | count 1
// PR2001

/* Test that the result of the assignment properly uses the value *in
   the bitfield* as opposed to the RHS. */
static int foo(int i) {
  struct {
    int f0 : 2;
  } x;
  return (x.f0 = i);
}

int bar() {
  return foo(-5) == -1;
}
