// RUN: %clang_cc1 -fblocks -fobjc-arc -fobjc-runtime-has-weak -g -triple x86_64-apple-darwin -emit-llvm %s -o - | FileCheck %s

// rdar://problem/14386148
// Test that the foo is aligned at an 8 byte boundary in the DWARF
// expression (256) that locates it inside of the byref descriptor:
// CHECK: [ DW_TAG_member ] [foo] [line 0, size {{[0-9]+}}, align 64, offset 256] [from Foo]

struct Foo {
  unsigned char *data;
};
int func() {
  __attribute__((__blocks__(byref))) struct Foo foo;
  return 0;
}
