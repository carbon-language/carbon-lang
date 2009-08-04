// RUN: clang-cc -triple x86_64-apple-darwin -frtti=0 -std=c++0x -S %s -o %t-64.s &&
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s &&
// RUN: clang-cc -triple i386-apple-darwin -frtti=0 -std=c++0x -S %s -o %t-32.s &&
// RUN: FileCheck -check-prefix LP32 -input-file=%t-32.s %s &&
// RUN: true

struct B {
  virtual void bar1();
  virtual void bar2();
};

static_assert (sizeof (B) == (sizeof(void *)), "vtable pointer layout");

class A : public B {
public:
  virtual void foo1();
  virtual void foo2();
  A() { }
} *a;

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

// CHECK-LP32: __ZTV1A:
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .long __ZN1B4bar1Ev
// CHECK-LP32: .long __ZN1B4bar2Ev
// CHECK-LP32: .long __ZN1A4foo1Ev
// CHECK-LP32: .long __ZN1A4foo2Ev
