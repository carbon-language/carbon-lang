// RUN: clang -fsyntax-only -verify %s

// PR3459
struct bar {
	char n[1];
};

struct foo {
	char name[(int)&((struct bar *)0)->n];
};
