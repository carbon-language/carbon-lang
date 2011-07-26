// RUN: %clang_cc1 %s -Os -emit-llvm -g -o - | grep DW_TAG_structure_type | count 1
// Variable 'a' is optimized but the debug info should preserve its type info.
#include <stdlib.h>

struct foo {
	int Attribute;
};

void *getfoo(void) __attribute__((noinline));

void *getfoo(void)
{
	int *x = malloc(sizeof(int));
	*x = 42;
	return (void *)x;
}

int main(int argc, char *argv[]) {
	struct foo *a = (struct foo *)getfoo();

	return a->Attribute;
}

