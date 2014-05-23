// RUN: %clang_cc1 -emit-llvm < %s > %t
// RUN: FileCheck %s --check-prefix=PRESENT < %t
// RUN: FileCheck %s --check-prefix=ABSENT  < %t

__attribute__((always_inline))
int test2() { return 0; }
// PRESENT-DAG: @test2{{.*}}[[ATTR2:#[0-9]+]]

__attribute__((optnone)) __attribute__((minsize))
int test3() { return 0; }
// PRESENT-DAG: @test3{{.*}}[[ATTR3:#[0-9]+]]

__attribute__((optnone)) __attribute__((cold))
int test4() { return test2(); }
// PRESENT-DAG: @test4{{.*}}[[ATTR4:#[0-9]+]]
// Also check that test2 is inlined into test4 (always_inline still works).
// PRESENT-NOT: call i32 @test2

// Check for both noinline and optnone on each optnone function.
// PRESENT-DAG: attributes [[ATTR3]] = { {{.*}}noinline{{.*}}optnone{{.*}} }
// PRESENT-DAG: attributes [[ATTR4]] = { {{.*}}noinline{{.*}}optnone{{.*}} }

// Check that no 'optsize' or 'minsize' attributes appear.
// ABSENT-NOT: optsize
// ABSENT-NOT: minsize
