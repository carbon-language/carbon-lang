// RUN: clang -emit-llvm %s
void f1()
{
	// Braces in a scalar
	int a = { 1 };
	int b = { 1, 2 };
}