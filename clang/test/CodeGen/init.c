// RUN: clang -emit-llvm %s
void f1()
{
	// Scalars in braces.
	int a = { 1 };
	int b = { 1, 2 };
}
