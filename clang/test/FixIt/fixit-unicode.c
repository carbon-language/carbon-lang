// RUN: %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck -strict-whitespace %s
// PR13312

struct Foo {
  int bar;
};

void test1() {
  struct Foo foo;
  (&foo)☃>bar = 42;
// CHECK: error: expected ';' after expression
// Make sure we emit the fixit right in front of the snowman.
// CHECK: {{^        \^}}
// CHECK: {{^        ;}}
}
