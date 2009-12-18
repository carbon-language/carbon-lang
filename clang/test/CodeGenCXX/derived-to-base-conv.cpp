// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++0x -S %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s
// RUN: %clang_cc1 -triple i386-apple-darwin -std=c++0x -S %s -o %t-32.s
// RUN: FileCheck -check-prefix LP32 --input-file=%t-32.s %s

extern "C" int printf(...);
extern "C" void exit(int);

struct A {
 A (const A&) { printf("A::A(const A&)\n"); }
 A() {};
}; 

struct B : public A {
  B() {};
}; 

struct C : public B {
  C() {};
}; 

struct X {
	operator B&() {printf("X::operator B&()\n"); return b; }
	operator C&() {printf("X::operator C&()\n"); return c; }
 	X (const X&) { printf("X::X(const X&)\n"); }
 	X () { printf("X::X()\n"); }
	B b;
	C c;
};

void f(A) {
  printf("f(A)\n");
}


void func(X x) 
{
  f (x);
}

int main()
{
    X x;
    func(x);
}

struct Base;

struct Root {
  operator Base&() { exit(1); }
};

struct Derived;

struct Base : Root {
  Base(const Base&) { printf("Base::(const Base&)\n"); }
  Base() { printf("Base::Base()\n"); }
  operator Derived&() { exit(1); }
};

struct Derived : Base {
};

void foo(Base) {}

void test(Derived bb)
{
	// CHECK-LP64-NOT: callq    __ZN4BasecvR7DerivedEv
	// CHECK-LP32-NOT: callq    L__ZN4BasecvR7DerivedEv
        foo(bb);
}
// CHECK-LP64: callq    __ZN1XcvR1BEv
// CHECK-LP64: callq    __ZN1AC1ERKS_

// CHECK-LP32: call     L__ZN1XcvR1BEv
// CHECK-LP32: call     L__ZN1AC1ERKS_


