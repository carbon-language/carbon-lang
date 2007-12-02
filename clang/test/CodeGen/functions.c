// RUN: clang %s -emit-llvm
int g();

int foo(int i) {
	return g(i);
}

int g(int i) {
	return g(i);
}

