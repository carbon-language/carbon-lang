// RUN: clang -emit-llvm %s
typedef short __v4hi __attribute__ ((__vector_size__ (8)));

void f()
{
    __v4hi A = (__v4hi)0LL;
}
