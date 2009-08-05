// RUN: clang-cc -triple x86_64-apple-darwin -frtti=0 -std=c++0x -S %s -o %t-64.s &&
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s &&
// RUN: clang-cc -triple i386-apple-darwin -frtti=0 -std=c++0x -S %s -o %t-32.s &&
// RUN: FileCheck -check-prefix LP32 -input-file=%t-32.s %s &&
// RUN: true

struct B {
  virtual void bar1();
  virtual void bar2();
};
void B::bar1() { }
void B::bar2() { }

struct C {
  virtual void bee1();
  virtual void bee2();
};
void C::bee1() { }
void C::bee2() { }

static_assert (sizeof (B) == (sizeof(void *)), "vtable pointer layout");

class A : public B, public C {
public:
  virtual void foo1();
  virtual void foo2();
  A() { }
} *a;
void A::foo1() { }
void A::foo2() { }

int main() {
  A a;
}

// CHECK-LP64: __ZTV1A:
// CHECK-LP64: .space 8
// CHECK-LP64: .space 8
// CHECK-LP64: .quad __ZN1B4bar1Ev
// CHECK-LP64: .quad __ZN1B4bar2Ev
// CHECK-LP64: .quad __ZN1A4foo1Ev
// CHECK-LP64: .quad __ZN1A4foo2Ev
// CHECK-LP64: .quad 18446744073709551608
// CHECK-LP64: .space 8
// CHECK-LP64: .quad __ZN1C4bee1Ev
// CHECK-LP64: .quad __ZN1C4bee2Ev

// CHECK-LP32: __ZTV1A:
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .long __ZN1B4bar1Ev
// CHECK-LP32: .long __ZN1B4bar2Ev
// CHECK-LP32: .long __ZN1A4foo1Ev
// CHECK-LP32: .long __ZN1A4foo2Ev
// CHECK-LP32: .long 4294967292
// CHECK-LP32: .space 4
// CHECK-LP32: .long __ZN1C4bee1Ev
// CHECK-LP32: .long __ZN1C4bee2Ev
