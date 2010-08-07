// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

struct A {
  virtual ~A();
};

struct B : A { };

struct C {
  int i;
  B b;
};

// CHECK: _Z15test_value_initv
void test_value_init() {
  // This value initialization requires zero initialization of the 'B'
  // subobject followed by a call to its constructor.
  // PR5800

  // CHECK: store i32 17
  // CHECK: call void @llvm.memset.p0i8.i64
  // CHECK: call void @_ZN1BC1Ev
  C c = { 17 } ;
  // CHECK: call void @_ZN1CD1Ev
}

enum enum_type { negative_number = -1, magic_number = 42 };

class enum_holder
{
  enum_type m_enum;

public:
  enum_holder() : m_enum(magic_number) { }
};

struct enum_holder_and_int
{
  enum_holder e;
  int i;
};

// CHECK: _Z24test_enum_holder_and_intv()
void test_enum_holder_and_int() {
  // CHECK: alloca
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: call void @llvm.memset
  // CHECK-NEXT: call void @_ZN19enum_holder_and_intC1Ev
  enum_holder_and_int();
  // CHECK-NEXT: ret void
}

// PR7834: don't crash.
namespace test1 {
  struct A {
    int A::*f;
    A();
    A(const A&);
    A &operator=(const A &);
  };

  struct B {
    A base;
  };

  void foo() {
    B();
  }
}
