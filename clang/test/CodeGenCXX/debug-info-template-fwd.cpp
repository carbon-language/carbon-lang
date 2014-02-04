// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -g -emit-llvm -o - | FileCheck %s
// This test is for a crash when emitting debug info for not-yet-completed types.
// Test that we don't actually emit a forward decl for the offending class:
// CHECK:  [ DW_TAG_class_type ] [Derived<Foo>] {{.*}} [def]
// rdar://problem/15931354
template <class R> class Returner {};
template <class A> class Derived;

template <class A>
class Base
{
  static Derived<A>* create();
};

template <class A>
class Derived : public Base<A> {
public:
  static void foo();
};

class Foo
{
  Foo();
  static Returner<Base<Foo> > all();
};

Foo::Foo(){}

Returner<Base<Foo> > Foo::all()
{
  Derived<Foo>::foo();
  return Foo::all();
}
