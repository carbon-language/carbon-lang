// RUN: clang-cc -triple x86_64-unknown-unknown -emit-llvm -o %t %s

// PR3442

static void *g(unsigned long len);

void
f(int n)
{
 unsigned begin_set[n];
 
 g(sizeof(begin_set));
}
