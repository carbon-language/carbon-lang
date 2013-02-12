// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -O2 -disable-llvm-optzns -o - %s | FileCheck %s

void test0(_Bool cond) {
  id test0_helper(void) __attribute__((ns_returns_retained));

  // CHECK:      define void @test0(
  // CHECK:      [[COND:%.*]] = alloca i8,
  // CHECK-NEXT: [[X:%.*]] = alloca i8*,
  // CHECK-NEXT: [[RELVAL:%.*]] = alloca i8*
  // CHECK-NEXT: [[RELCOND:%.*]] = alloca i1
  // CHECK-NEXT: zext
  // CHECK-NEXT: store
  // CHECK-NEXT: [[T0:%.*]] = load i8* [[COND]]
  // CHECK-NEXT: [[T1:%.*]] = trunc i8 [[T0]] to i1
  // CHECK-NEXT: store i1 false, i1* [[RELCOND]]
  // CHECK-NEXT: br i1 [[T1]],
  // CHECK:      br label
  // CHECK:      [[CALL:%.*]] = call i8* @test0_helper()
  // CHECK-NEXT: store i8* [[CALL]], i8** [[RELVAL]]
  // CHECK-NEXT: store i1 true, i1* [[RELCOND]]
  // CHECK-NEXT: br label
  // CHECK:      [[T0:%.*]] = phi i8* [ null, {{%.*}} ], [ [[CALL]], {{%.*}} ]
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retain(i8* [[T0]]) nounwind
  // CHECK-NEXT: store i8* [[T1]], i8** [[X]],
  // CHECK-NEXT: [[REL:%.*]] = load i1* [[RELCOND]]
  // CHECK-NEXT: br i1 [[REL]],
  // CHECK:      [[T0:%.*]] = load i8** [[RELVAL]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]]) nounwind
  // CHECK-NEXT: br label
  // CHECK:      [[T0:%.*]] = load i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]]) nounwind
  // CHECK-NEXT: ret void
  id x = (cond ? 0 : test0_helper());
}

void test1(int cond) {
  __strong id strong;
  __weak id weak;
  extern void test1_sink(id *);
  test1_sink(cond ? &strong : 0);
  test1_sink(cond ? &weak : 0);

  // CHECK:    define void @test1(
  // CHECK:      [[COND:%.*]] = alloca i32
  // CHECK-NEXT: [[STRONG:%.*]] = alloca i8*
  // CHECK-NEXT: [[WEAK:%.*]] = alloca i8*
  // CHECK-NEXT: [[TEMP1:%.*]] = alloca i8*
  // CHECK-NEXT: [[TEMP2:%.*]] = alloca i8*
  // CHECK-NEXT: [[CONDCLEANUPSAVE:%.*]] = alloca i8*
  // CHECK-NEXT: [[CONDCLEANUP:%.*]] = alloca i1
  // CHECK-NEXT: store i32
  // CHECK-NEXT: store i8* null, i8** [[STRONG]]
  // CHECK-NEXT: call i8* @objc_initWeak(i8** [[WEAK]], i8* null)

  // CHECK-NEXT: [[T0:%.*]] = load i32* [[COND]]
  // CHECK-NEXT: [[T1:%.*]] = icmp ne i32 [[T0]], 0
  // CHECK:      [[ARG:%.*]] = phi i8**
  // CHECK-NEXT: [[T0:%.*]] = icmp eq i8** [[ARG]], null
  // CHECK-NEXT: [[T1:%.*]] = select i1 [[T0]], i8** null, i8** [[TEMP1]]
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = load i8** [[ARG]]
  // CHECK-NEXT: store i8* [[T0]], i8** [[TEMP1]]
  // CHECK-NEXT: br label
  // CHECK:      call void @test1_sink(i8** [[T1]])
  // CHECK-NEXT: [[T0:%.*]] = icmp eq i8** [[ARG]], null
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = load i8** [[TEMP1]]
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retain(i8* [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = load i8** [[ARG]]
  // CHECK-NEXT: store i8* [[T1]], i8** [[ARG]]
  // CHECK-NEXT: call void @objc_release(i8* [[T2]])
  // CHECK-NEXT: br label

  // CHECK:      [[T0:%.*]] = load i32* [[COND]]
  // CHECK-NEXT: [[T1:%.*]] = icmp ne i32 [[T0]], 0
  // CHECK:      [[ARG:%.*]] = phi i8**
  // CHECK-NEXT: [[T0:%.*]] = icmp eq i8** [[ARG]], null
  // CHECK-NEXT: [[T1:%.*]] = select i1 [[T0]], i8** null, i8** [[TEMP2]]
  // CHECK-NEXT: store i1 false, i1* [[CONDCLEANUP]]
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = call i8* @objc_loadWeakRetained(i8** [[ARG]])
  // CHECK-NEXT: store i8* [[T0]], i8** [[CONDCLEANUPSAVE]]
  // CHECK-NEXT: store i1 true, i1* [[CONDCLEANUP]]
  // CHECK-NEXT: store i8* [[T0]], i8** [[TEMP2]]
  // CHECK-NEXT: br label
  // CHECK:      call void @test1_sink(i8** [[T1]])
  // CHECK-NEXT: [[T0:%.*]] = icmp eq i8** [[ARG]], null
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = load i8** [[TEMP2]]
  // CHECK-NEXT: call i8* @objc_storeWeak(i8** [[ARG]], i8* [[T0]])
  // CHECK-NEXT: br label

  // CHECK:      call void @objc_destroyWeak(i8** [[WEAK]])
  // CHECK:      ret void
}
