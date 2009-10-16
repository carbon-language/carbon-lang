// RUN: clang-cc -triple x86_64-apple-darwin -std=c++0x -S %s -o %t-64.s &&
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s &&
// RUN: clang-cc -triple i386-apple-darwin -std=c++0x -S %s -o %t-32.s &&
// RUN: FileCheck -check-prefix LP32 --input-file=%t-32.s %s &&
// RUN: true

extern "C" int printf(...);

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

// CHECK-LP64: call     __ZN1XcvR1BEv
// CHECK-LP64: call     __ZN1AC1ERKS_

// CHECK-LP32: call     L__ZN1XcvR1BEv
// CHECK-LP32: call     L__ZN1AC1ERKS_
