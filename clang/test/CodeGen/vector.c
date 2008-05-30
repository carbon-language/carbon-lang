// RUN: clang -emit-llvm %s
typedef short __v4hi __attribute__ ((__vector_size__ (8)));

void f()
{
    __v4hi A = (__v4hi)0LL;
}

__v4hi x = {1,2,3};
__v4hi y = {1,2,3,4};
__v4hi z = {1,2,3,4,5};
