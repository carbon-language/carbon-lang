// RUN: %llvmgcc %s -S -O1 -o - | llvm-as | opt -std-compile-opts | llvm-dis | not grep add

struct S { int A; int B; char C[1000]; };

int f(struct S x) __attribute__ ((const));

static int __attribute__ ((const)) g(struct S x) {
	x.A = x.B;
	return f(x);
}

int h(void) {
	struct S x;
	int r;
	x.A = 0;
	x.B = 9;
	r = g(x);
	return r + x.A;
}
