// RUN: %clang_cc1 %s -emit-llvm-only -verify
// PR7291

struct Foo {
  unsigned file_id;

  Foo(unsigned arg);
};

Foo::Foo(unsigned arg) : file_id(arg = 42)
{ }

