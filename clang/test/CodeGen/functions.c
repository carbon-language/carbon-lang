// RUN: clang-cc %s -emit-llvm -o %t &&

int g();

int foo(int i) {
	return g(i);
}

int g(int i) {
	return g(i);
}

// rdar://6110827
typedef void T(void);
void test3(T f) {
  f();
}

int a(int);
int a() {return 1;}

// RUN: grep 'define void @f0()' %t &&
void f0() {}

void f1();
// RUN: grep 'call void @f1()' %t &&
void f2(void) {
  f1(1, 2, 3);
}
// RUN: grep 'define void @f1()' %t &&
void f1() {}

// RUN: grep 'define .* @f3' %t | not grep -F '...'
struct foo { int X, Y, Z; } f3() {
}
