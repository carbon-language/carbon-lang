// RUN: clang-cc %s -emit-llvm -o %t &&
struct S {
	operator int();
};

// RUN: grep "_ZN1ScviEv" %t
S::operator int() {
	return 10;
}
