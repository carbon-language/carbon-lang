// RUN: clang %s -emit-llvm -o -
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

