// RUN: clang-cc %s -emit-llvm -o %t &&
// RUN: grep 'call void @_ZN1XC1ERK1Xiii' %t | count 3

extern "C" int printf(...);


struct C {
	C() : iC(6) {}
	int iC;
};

int foo() {
  return 6;
};

class X { // ...
public: 
	X(int) {}
	X(const X&, int i = 1, int j = 2, int k = foo()) {
		printf("X(const X&, %d, %d, %d)\n", i, j, k);
	}
};

int main()
{
	X a(1);
	X b(a, 2);
	X c = b;
	X d(a, 5, 6);
}
