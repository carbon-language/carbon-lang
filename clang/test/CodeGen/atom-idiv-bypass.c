// This test verifies that IDIV bypass optimizations is used when compiling for
// Atom processors with O2. The optimization inserts a test followed by a
// branch to bypass the slow IDIV instruction. This test verifies that global
// context types are used when comparing with those in the BypassTypeMap.

// RUN: %clang %s -triple i386-unknown-unknown -march=atom -m32 -O2 -S -o - | FileCheck %s
// CHECK: div32
// CHECK: orl
// CHECK: testl
// CHECK: je
// CHECK: idivl
// CHECK: ret
// CHECK: divb
// CHECK: ret
int div32(int a, int b)
{
  return a/b;
}

// CHECK: divrem32
// CHECK: orl
// CHECK: testl
// CHECK: je
// CHECK: idivl
// CHECK: jmp
// CHECK: divb
// CHECK: addl
// CHECK: ret
int divrem32(int a, int b)
{
  return a/b + a%b;
}
