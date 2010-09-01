// RUN: %llvmgcc %s -m32 -S -o - | FileCheck %s
// Misaligned parameter must be memcpy'd to correctly aligned temporary.
// XFAIL: *
// XTARGET: i386-apple-darwin,i686-apple-darwin,x86_64-apple-darwin

struct s { int x; long double y; };
long double foo(struct s x, int i, struct s y) {
// CHECK: foo
// CHECK: %x_addr = alloca %struct.s, align 16
// CHECK: %y_addr = alloca %struct.s, align 16
// CHECK: memcpy
// CHECK: memcpy
// CHECK: bar
  return bar(&x, &y);
}
