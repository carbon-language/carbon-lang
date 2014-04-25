// RUN: clang %s -O3 -target arm64-apple-ios -o - -S -mllvm -debug-only=loop-reduce 2>&1| FileCheck %s
// REQUIRES: asserts

// LSR used to fail here due to a bug in the ReqRegs test.  To complicate
// things, this could only be reproduced with clang because the uses would
// come out in different order when invoked through llc.

// CHECK: The chosen solution requires
// CHECK-NOT: No Satisfactory Solution

typedef unsigned long iter_t;
void use_int(int result);

struct _state {
 int N;
 int M;
 int K;
 double* data;
};
void
do_integer_add(iter_t iterations, void* cookie)
{
    struct _state *pState = (struct _state*)cookie;
    register int i;
    register int a = pState->N + 57;

    while (iterations-- > 0) {
        for (i = 1; i < 1001; ++i) {
          a=a+a+i; a=a+a+i; a=a+a+i; a=a+a+i;
          a=a+a+i; a=a+a+i; a=a+a+i; a=a+a+i;
          a=a+a+i; a=a+a+i;

        }
    }
    use_int(a);
}
