// RUN: clang -fsyntax-only -Wunused-variable -verify %s

template<typename T> void f() {
	T t;
	t = 17;
}
