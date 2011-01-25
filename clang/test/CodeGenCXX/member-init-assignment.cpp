// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s
// PR7291

struct Foo {
  unsigned file_id;

  Foo(unsigned arg);
};

Foo::Foo(unsigned arg) : file_id(arg = 42)
{ }

// CHECK: define void @_ZN3FooC2Ej(%struct.Foo* %this, i32 %arg) unnamed_addr
// CHECK: [[ARG:%.*]] = alloca i32
// CHECK: store i32 42, i32* [[ARG]]
// CHECK: store i32 42, i32* %{{.*}}
// CHECK: ret void
