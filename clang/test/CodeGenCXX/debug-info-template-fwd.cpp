// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -g -emit-llvm -o - | FileCheck %s
// This test is for a crash when emitting debug info for not-yet-completed types.
// Test that we don't actually emit a forward decl for the offending class:
// CHECK:  [ DW_TAG_class_type ] [Derived<const __CFString, Foo>] {{.*}} [def]
// rdar://problem/15931354
typedef const struct __CFString * CFStringRef;
template <class R> class Returner {};
typedef const __CFString String;

template <class A, class B> class Derived;

template <class A, class B>
class Base
{
  static Derived<A, B>* create();
};

template <class A, class B>
class Derived : public Base<A, B> {
public:
  static void foo();
};

class Foo
{
  Foo();
  static Returner<Base<String,Foo> > all();
};

Foo::Foo(){}

Returner<Base<String,Foo> > Foo::all()
{
  Derived<String,Foo>::foo();
  return Foo::all();
}
