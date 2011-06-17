// RUN: %clang_cc1 -fblocks -fobjc-arc -fobjc-nonfragile-abi -triple x86_64-apple-darwin -O0 -emit-llvm %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s
// rdar://9503326
// rdar://9606600

extern void use(id);
extern void use_block(void (^)(void));
@class NSArray;

void test0(NSArray *array) {
  // 'x' should be initialized without a retain.
  // We should actually do a non-constant capture, and that
  // capture should require a retain.
  for (id x in array) {
    use_block(^{ use(x); });
  }
}

// CHECK-LP64:    define void @test0(
// CHECK-LP64:      alloca [[ARRAY_T:%.*]]*,
// CHECK-LP64-NEXT: [[X:%.*]] = alloca i8*,
// CHECK-LP64-NEXT: [[STATE:%.*]] = alloca [[STATE_T:%.*]],
// CHECK-LP64-NEXT: alloca [16 x i8*], align 8
// CHECK-LP64-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:%.*]],

// CHECK-LP64:      [[T0:%.*]] = getelementptr inbounds [[STATE_T]]* [[STATE]], i32 0, i32 1
// CHECK-LP64-NEXT: [[T1:%.*]] = load i8*** [[T0]]
// CHECK-LP64-NEXT: [[T2:%.*]] = getelementptr i8** [[T1]], i64
// CHECK-LP64-NEXT: [[T3:%.*]] = load i8** [[T2]]
// CHECK-LP64-NEXT: store i8* [[T3]], i8** [[X]]

// CHECK-LP64:      [[T0:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-LP64-NEXT: [[T1:%.*]] = load i8** [[X]]
// CHECK-LP64-NEXT: [[T2:%.*]] = call i8* @objc_retain(i8* [[T1]])
// CHECK-LP64-NEXT: store i8* [[T2]], i8** [[T0]]
// CHECK-LP64-NEXT: [[T1:%.*]] = bitcast [[BLOCK_T]]* [[BLOCK]] to void ()*
// CHECK-LP64-NEXT: call void @use_block(void ()* [[T1]])
// CHECK-LP64-NEXT: [[T1:%.*]] = load i8** [[T0]]
// CHECK-LP64-NEXT: call void @objc_release(i8* [[T1]])

// CHECK-LP64:    define internal void @__test0_block_invoke
// CHECK-LP64:      [[BLOCK:%.*]] = bitcast i8* {{%.*}} to [[BLOCK_T]]*
// CHECK-LP64-NEXT: [[T0:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-LP64-NEXT: [[T2:%.*]] = load i8** [[T0]], align 8 
// CHECK-LP64-NEXT: call void @use(i8* [[T2]])

void test1(NSArray *array) {
  for (__weak id x in array) {
    use_block(^{ use(x); });
  }
}

// CHECK-LP64:    define void @test1(
// CHECK-LP64:      alloca [[ARRAY_T:%.*]]*,
// CHECK-LP64-NEXT: [[X:%.*]] = alloca i8*,
// CHECK-LP64-NEXT: [[STATE:%.*]] = alloca [[STATE_T:%.*]],
// CHECK-LP64-NEXT: alloca [16 x i8*], align 8
// CHECK-LP64-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:%.*]],

// CHECK-LP64:      [[T0:%.*]] = getelementptr inbounds [[STATE_T]]* [[STATE]], i32 0, i32 1
// CHECK-LP64-NEXT: [[T1:%.*]] = load i8*** [[T0]]
// CHECK-LP64-NEXT: [[T2:%.*]] = getelementptr i8** [[T1]], i64
// CHECK-LP64-NEXT: [[T3:%.*]] = load i8** [[T2]]
// CHECK-LP64-NEXT: call i8* @objc_initWeak(i8** [[X]], i8* [[T3]])

// CHECK-LP64:      [[T0:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-LP64-NEXT: [[T1:%.*]] = call i8* @objc_loadWeak(i8** [[X]])
// CHECK-LP64-NEXT: call i8* @objc_initWeak(i8** [[T0]], i8* [[T1]])
// CHECK-LP64-NEXT: [[T1:%.*]] = bitcast [[BLOCK_T]]* [[BLOCK]] to void ()*
// CHECK-LP64-NEXT: call void @use_block(void ()* [[T1]])
// CHECK-LP64-NEXT: call void @objc_destroyWeak(i8** [[T0]])
// CHECK-LP64-NEXT: call void @objc_destroyWeak(i8** [[X]])
