// RUN: %clang_cc1 -g -emit-llvm -o - %s | FileCheck %s

// Make sure that clang outputs distinct debug info for a function
// that is inlined twice on the same line. Otherwise it would appear
// as if the function was only inlined once.

#define INLINE inline __attribute__((always_inline))

INLINE int
product (int x, int y)
{
    int result = x * y;
    return result;
}

INLINE int
sum (int a, int b)
{
    int result = a + b;
    return result;
}

int
strange_max (int m, int n)
{
    if (m > n)
        return m;
    else if (n > m)
        return n;
    else
        return 0;
}

int
foo (int i, int j)
{
    if (strange_max (i, j) == i)
        return product (i, j);
    else if (strange_max  (i, j) == j)
        return sum (i, j);
    else
        return product (sum (i, i), sum (j, j));
}

int
main(int argc, char const *argv[])
{

    int array[3];
    int n;

    array[0] = foo (1238, 78392);
    array[1] = foo (379265, 23674);
    array[2] = foo (872934, 234);

    n = strange_max(array[0], strange_max(array[1], array[2]));

    return n & 0xf;
}

// CHECK: define {{.*}} @_Z3fooii
// i
// CHECK: call void @llvm.dbg.declare
// j
// CHECK: call void @llvm.dbg.declare
// x
// CHECK: call void @llvm.dbg.declare
// y
// CHECK: call void @llvm.dbg.declare
// result
// CHECK: call void @llvm.dbg.declare

// CHECK: call void @llvm.dbg.declare(metadata !{i32* %{{.*}}}, metadata ![[A_MD:[0-9]+]]), !dbg ![[A_DI:[0-9]+]]
// CHECK: call void @llvm.dbg.declare(metadata !{i32* %{{.*}}}, metadata ![[B_MD:[0-9]+]]), !dbg ![[B_DI:[0-9]+]]
// result
// CHECK: call void @llvm.dbg.declare

// We want to see a distinct !dbg node.
// CHECK-NOT: call void @llvm.dbg.declare(metadata !{i32* %{{.*}}}, metadata ![[A_MD]]), !dbg ![[A_DI]]
// CHECK:     call void @llvm.dbg.declare(metadata !{i32* %{{.*}}}, metadata ![[A_MD]]), !dbg !{{.*}}
// CHECK-NOT: call void @llvm.dbg.declare(metadata !{i32* %{{.*}}}, metadata ![[B_MD]]), !dbg ![[B_DI]]
// CHECK:     call void @llvm.dbg.declare(metadata !{i32* %{{.*}}}, metadata ![[B_MD]]), !dbg !{{.*}}
// result
// CHECK: call void @llvm.dbg.declare

// CHECK: define {{.*}} @main
// CHECK: call {{.*}} @_Z3fooii
// CHECK: call {{.*}} @_Z3fooii
// CHECK: call {{.*}} @_Z3fooii
// CHECK: store
// CHECK: getelementptr
// We want to see the same !dbg node for non-inlined functions. 
// Needed for GDB compatibility.
// CHECK: load {{.*}} !dbg ![[DBG:.*]]
// CHECK: load {{.*}} !dbg ![[DBG]]
// CHECK: load {{.*}} !dbg ![[DBG]]
// CHECK: call {{.*}} @_Z11strange_maxii(i32 %{{.*}}, i32 %{{.*}}), !dbg ![[DBG]]
// CHECK: call {{.*}} @_Z11strange_maxii(i32 %{{.*}}, i32 %{{.*}}), !dbg ![[DBG]]
