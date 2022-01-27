// RUN: %clang_cc1 -triple amdgcn---amdgiz -emit-llvm < %s | FileCheck -check-prefixes=CHECK %s

// CHECK-DAG: @foo ={{.*}} addrspace(1) global i32 0
int foo;

// CHECK-DAG: @ban ={{.*}} addrspace(1) global [10 x i32] zeroinitializer
int ban[10];

// CHECK-DAG: @A ={{.*}} addrspace(1) global i32* null
// CHECK-DAG: @B ={{.*}} addrspace(1) global i32* null
int *A;
int *B;

// CHECK-LABEL: define{{.*}} i32 @test1()
// CHECK: load i32, i32* addrspacecast{{[^@]+}} @foo
int test1() { return foo; }

// CHECK-LABEL: define{{.*}} i32 @test2(i32 noundef %i)
// CHECK: %[[addr:.*]] = getelementptr
// CHECK: load i32, i32* %[[addr]]
// CHECK-NEXT: ret i32
int test2(int i) { return ban[i]; }

// CHECK-LABEL: define{{.*}} void @test3()
// CHECK: load i32*, i32** addrspacecast{{.*}} @B
// CHECK: load i32, i32*
// CHECK: load i32*, i32** addrspacecast{{.*}} @A
// CHECK: store i32 {{.*}}, i32*
void test3() {
  *A = *B;
}

// CHECK-LABEL: define{{.*}} void @test4(i32* noundef %a)
// CHECK: %[[alloca:.*]] = alloca i32*, align 8, addrspace(5)
// CHECK: %[[a_addr:.*]] ={{.*}} addrspacecast{{.*}} %[[alloca]] to i32**
// CHECK: store i32* %a, i32** %[[a_addr]]
// CHECK: %[[r0:.*]] = load i32*, i32** %[[a_addr]]
// CHECK: %[[arrayidx:.*]] = getelementptr inbounds i32, i32* %[[r0]]
// CHECK: store i32 0, i32* %[[arrayidx]]
void test4(int *a) {
  a[0] = 0;
}
