// RUN: clang -fsyntax-only -verify %s

// PR3459
struct bar {
	char n[1];
};

struct foo {
	char name[(int)&((struct bar *)0)->n];
	char name2[(int)&((struct bar *)0)->n - 1]; //expected-error{{fields must have a constant size}}
};
