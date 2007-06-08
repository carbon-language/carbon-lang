// RUN: clang %s -fsyntax-only

void test1(int n, int* p) { *(n ? p : (void *)(7-7)) = 1; }
void test2(int n, int* p) { *(n ? p : (void *)0) = 1; }

