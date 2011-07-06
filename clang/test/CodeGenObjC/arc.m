// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-nonfragile-abi -fblocks -fobjc-arc -fobjc-runtime-has-weak -O2 -disable-llvm-optzns -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-nonfragile-abi -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck -check-prefix=CHECK-GLOBALS %s

// CHECK: define void @test0
void test0(id x) {
  // CHECK:      [[X:%.*]] = alloca i8*
  // CHECK-NEXT: [[PARM:%.*]] = call i8* @objc_retain(i8* {{.*}})
  // CHECK-NEXT: store i8* [[PARM]], i8** [[X]]
  // CHECK-NEXT: [[TMP:%.*]] = load i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[TMP]])
  // CHECK-NEXT: ret void
}

// CHECK: define i8* @test1(i8*
id test1(id x) {
  // CHECK:      [[RET:%.*]] = alloca i8*
  // CHECK-NEXT: [[X:%.*]] = alloca i8*
  // CHECK-NEXT: [[Y:%.*]] = alloca i8*
  // CHECK-NEXT: alloca i32
  // CHECK-NEXT: [[PARM:%.*]] = call i8* @objc_retain(i8* {{%.*}})
  // CHECK-NEXT: store i8* [[PARM]], i8** [[X]]
  // CHECK-NEXT: store i8* null, i8** [[Y]]
  // CHECK-NEXT: [[T0:%.*]] = load i8** [[Y]]
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retain(i8* [[T0]])
  // CHECK-NEXT: store i8* [[T1]], i8** [[RET]]
  // CHECK-NEXT: store i32
  // CHECK-NEXT: [[T0:%.*]] = load i8** [[Y]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]])
  // CHECK-NEXT: [[T1:%.*]] = load i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])
  // CHECK-NEXT: [[T0:%.*]] = load i8** [[RET]]
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_autoreleaseReturnValue(i8* [[T0]])
  // CHECK-NEXT: ret i8* [[T1]]
  id y;
  return y;
}

@interface Test2
+ (void) class_method;
- (void) inst_method;
@end
@implementation Test2

// The self pointer of a class method is not retained.
// CHECK: define internal void @"\01+[Test2 class_method]"
// CHECK:      alloca
// CHECK-NEXT: alloca
// CHECK-NEXT: store
// CHECK-NEXT: store
// CHECK-NEXT: ret void
+ (void) class_method {}

// The self pointer of an instance method is not retained.
// CHECK: define internal void @"\01-[Test2 inst_method]"
// CHECK:      alloca
// CHECK-NEXT: alloca
// CHECK-NEXT: store
// CHECK-NEXT: store
// CHECK-NEXT: ret void
- (void) inst_method {}
@end

@interface Test3
+ (id) alloc;
- (id) initWith: (int) x;
- (id) copy;
@end

// CHECK: define void @test3_unelided()
void test3_unelided() {
  extern void test3_helper(void);

  // CHECK:      [[X:%.*]] = alloca [[TEST3:%.*]]*
  // CHECK-NEXT: store [[TEST3]]* null, [[TEST3]]** [[X]], align
  Test3 *x;

  // Call to +alloc.
  // CHECK-NEXT: load {{.*}}* @"\01L_OBJC_CLASSLIST_REFERENCES_
  // CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: [[ALLOC:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: call void @objc_release(i8*
  [Test3 alloc];

  // CHECK-NEXT: [[T0:%.*]] = load [[TEST3]]** [[X]]
  // CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST3]]* [[T0]] to i8*
  // CHECK-NEXT: [[COPY:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend {{.*}})(i8* [[T1]],
  // CHECK-NEXT: call void @objc_release(i8* [[COPY]]) nounwind
  [x copy];

  // CHECK-NEXT: [[T0:%.*]] = load [[TEST3]]** [[X]]
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST3]]* [[T0]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T1]]) nounwind
  // CHECK-NEXT: ret void
}

// CHECK: define void @test3()
void test3() {
  // CHECK:      [[X:%.*]] = alloca i8*

  id x = [[Test3 alloc] initWith: 5];

  // Call to +alloc.
  // CHECK-NEXT: load {{.*}}* @"\01L_OBJC_CLASSLIST_REFERENCES_
  // CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: [[ALLOC:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
  // CHECK-NEXT: bitcast

  // Call to -initWith: with elided retain of consumed argument.
  // CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: [[INIT:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i32)*)(i8* 
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: [[INIT:%.*]] = bitcast
  // Assignment for initialization, retention elided.
  // CHECK-NEXT: store i8* [[INIT]], i8** [[X]]

  // Call to -copy.
  // CHECK-NEXT: [[V:%.*]] = load i8** [[X]]
  // CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[COPY:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend {{.*}})(i8* [[V]],

  // Assignment to x.
  // CHECK-NEXT: [[TMP:%.*]] = load i8** [[X]]
  // CHECK-NEXT: store i8* [[COPY]], i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[TMP]]) nounwind

  x = [x copy];

  // Cleanup for x.
  // CHECK-NEXT: [[TMP:%.*]] = load i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[TMP]]) nounwind
  
  // CHECK-NEXT: ret void
}

// CHECK: define i8* @test4()
id test4() {
  // Call to +alloc.
  // CHECK:      load {{.*}}* @"\01L_OBJC_CLASSLIST_REFERENCES_
  // CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: [[ALLOC:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
  // CHECK-NEXT: [[ALLOC:%.*]] = bitcast

  // Call to -initWith: with elided retain of consumed argument.
  // CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[ALLOC:%.*]] = bitcast
  // CHECK-NEXT: [[INIT:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i32)*)(i8* [[ALLOC]],

  // Initialization of return value, occuring within full-expression.
  // Retain/release elided.
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: [[INIT:%.*]] = bitcast
  // CHECK-NEXT: [[RET:%.*]] = call i8* @objc_autoreleaseReturnValue(i8* [[INIT]])

  // CHECK-NEXT: ret i8* [[RET]]

  return [[Test3 alloc] initWith: 6];
}

@interface Test5 {
@public
  id var;
}
@end

// CHECK: define void @test5
void test5(Test5 *x, id y) {
  // Prologue.
  // CHECK:      [[X:%.*]] = alloca [[TEST5:%.*]]*,
  // CHECK-NEXT: [[Y:%.*]] = alloca i8*
  // CHECK-NEXT: bitcast [[TEST5]]* {{%.*}} to i8*
  // CHECK-NEXT: call i8* @objc_retain
  // CHECK-NEXT: [[PARMX:%.*]] = bitcast i8* {{%.*}} to [[TEST5]]*
  // CHECK-NEXT: store [[TEST5]]* [[PARMX]], [[TEST5]]** [[X]]
  // CHECK-NEXT: call i8* @objc_retain
  // CHECK-NEXT: store

  // CHECK-NEXT: load [[TEST5]]** [[X]]
  // CHECK-NEXT: load i64* @"OBJC_IVAR_$_Test5.var"
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: [[VAR:%.*]] = bitcast
  // CHECK-NEXT: [[TMP:%.*]] = load i8** [[VAR]]
  // CHECK-NEXT: store i8* null, i8** [[VAR]]
  // CHECK-NEXT: call void @objc_release(i8* [[TMP]]) nounwind
  x->var = 0;

  // CHECK-NEXT: [[YVAL:%.*]] = load i8** [[Y]]
  // CHECK-NEXT: load [[TEST5]]** [[X]]
  // CHECK-NEXT: load i64* @"OBJC_IVAR_$_Test5.var"
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: [[VAR:%.*]] = bitcast
  // CHECK-NEXT: [[T0:%.*]] = call i8* @objc_retain(i8* [[YVAL]]) nounwind
  // CHECK-NEXT: [[TMP:%.*]] = load i8** [[VAR]]
  // CHECK-NEXT: store i8* [[T0]], i8** [[VAR]]
  // CHECK-NEXT: call void @objc_release(i8* [[TMP]]) nounwind
  x->var = y;

  // Epilogue.
  // CHECK-NEXT: [[TMP:%.*]] = load i8** [[Y]]
  // CHECK-NEXT: call void @objc_release(i8* [[TMP]]) nounwind
  // CHECK-NEXT: [[T0:%.*]] = load [[TEST5]]** [[X]]
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST5]]* [[T0]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T1]]) nounwind
  // CHECK-NEXT: ret void
}

id test6_helper(void) __attribute__((ns_returns_retained));
// CHECK: define void @test6()
void test6() {
  // CHECK:      [[X:%.*]] = alloca i8*
  // CHECK-NEXT: [[CALL:%.*]] = call i8* @test6_helper()
  // CHECK-NEXT: store i8* [[CALL]], i8** [[X]]
  // CHECK-NEXT: [[T1:%.*]] = load i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T1]]) nounwind, !clang.imprecise_release
  // CHECK-NEXT: ret void
  id x = test6_helper();
}

void test7_helper(id __attribute__((ns_consumed)));
// CHECK: define void @test7()
void test7() {
  // CHECK:      [[X:%.*]] = alloca i8*
  // CHECK-NEXT: store i8* null, i8** [[X]]
  // CHECK-NEXT: [[T0:%.*]] = load i8** [[X]]
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retain(i8* [[T0]]) nounwind
  // CHECK-NEXT: call void @test7_helper(i8* [[T1]])
  // CHECK-NEXT: [[T1:%.*]] = load i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T1]]) nounwind, !clang.imprecise_release
  // CHECK-NEXT: ret void
  id x;
  test7_helper(x);
}

id test8_helper(void) __attribute__((ns_returns_retained));
void test8() {
  __unsafe_unretained id x = test8_helper();
  // CHECK:      [[X:%.*]] = alloca i8*
  // CHECK-NEXT: [[T0:%.*]] = call i8* @test8_helper()
  // CHECK-NEXT: store i8* [[T0]], i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]]) nounwind
  // CHECK-NOT:  imprecise_release
  // CHECK-NEXT: ret void
}

id test9_helper(void) __attribute__((ns_returns_retained));
void test9() {
  id x __attribute__((objc_precise_lifetime)) = test9_helper();
  x = 0;
  // CHECK:      [[X:%.*]] = alloca i8*
  // CHECK-NEXT: [[CALL:%.*]] = call i8* @test9_helper()
  // CHECK-NEXT: store i8* [[CALL]], i8** [[X]]

  // CHECK-NEXT: [[T1:%.*]] = load i8** [[X]]
  // CHECK-NEXT: store i8* null, i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T1]]) nounwind, !clang.imprecise_release

  // CHECK-NEXT: [[T1:%.*]] = load i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T1]]) nounwind
  // CHECK-NOT:  clang.imprecise_release

  // CHECK-NEXT: ret void
}

@interface Test10
@property (retain) Test10 *me;
@end
void test10() {
  Test10 *x;
  id y = x.me.me;

  // CHECK:      define void @test10()
  // CHECK:      [[X:%.*]] = alloca [[TEST10:%.*]]*, align
  // CHECK-NEXT: [[Y:%.*]] = alloca i8*, align
  // CHECK-NEXT: store [[TEST10]]* null, [[TEST10]]** [[X]]
  // CHECK-NEXT: load [[TEST10]]** [[X]], align
  // CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES_{{[0-9]*}}"
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: call [[TEST10]]* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
  // CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES_{{[0-9]*}}"
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: [[T0:%.*]] = call [[TEST10]]* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST10]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[TEST10]]*
  // CHECK-NEXT: [[T4:%.*]] = bitcast [[TEST10]]* [[T3]] to i8*
  // CHECK-NEXT: store i8* [[T4]], i8** [[Y]]
  // CHECK-NEXT: [[T0:%.*]] = load i8** [[Y]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]])
  // CHECK-NEXT: [[T0:%.*]] = load [[TEST10]]** [[X]]
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST10]]* [[T0]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])
  // CHECK-NEXT: ret void
}

void test11(id (*f)(void) __attribute__((ns_returns_retained))) {
  // CHECK:      define void @test11(
  // CHECK:      [[F:%.*]] = alloca i8* ()*, align
  // CHECK-NEXT: [[X:%.*]] = alloca i8*, align
  // CHECK-NEXT: store i8* ()* {{%.*}}, i8* ()** [[F]], align
  // CHECK-NEXT: [[T0:%.*]] = load i8* ()** [[F]], align
  // CHECK-NEXT: [[T1:%.*]] = call i8* [[T0]]()
  // CHECK-NEXT: store i8* [[T1]], i8** [[X]], align
  // CHECK-NEXT: [[T3:%.*]] = load i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T3]]) nounwind, !clang.imprecise_release
  // CHECK-NEXT: ret void
  id x = f();
}

void test12(void) {
  extern id test12_helper(void);

  // CHECK:      define void @test12()
  // CHECK:      [[X:%.*]] = alloca i8*, align
  // CHECK-NEXT: [[Y:%.*]] = alloca i8*, align

  __weak id x = test12_helper();
  // CHECK-NEXT: [[T0:%.*]] = call i8* @test12_helper()
  // CHECK-NEXT: call i8* @objc_initWeak(i8** [[X]], i8* [[T0]])

  x = test12_helper();
  // CHECK-NEXT: [[T1:%.*]] = call i8* @test12_helper()
  // CHECK-NEXT: call i8* @objc_storeWeak(i8** [[X]], i8* [[T1]])

  id y = x;
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_loadWeakRetained(i8** [[X]])
  // CHECK-NEXT: store i8* [[T2]], i8** [[Y]], align

  // CHECK-NEXT: [[T4:%.*]] = load i8** [[Y]]
  // CHECK-NEXT: call void @objc_release(i8* [[T4]]) nounwind, !clang.imprecise_release
  // CHECK-NEXT: call void @objc_destroyWeak(i8** [[X]])
  // CHECK-NEXT: ret void
}

// Indirect consuming calls.
void test13(void) {
  // CHECK:      define void @test13()
  // CHECK:      [[X:%.*]] = alloca i8*, align
  // CHECK-NEXT: store i8* null, i8** [[X]], align
  id x;

  typedef void fnty(id __attribute__((ns_consumed)));
  extern fnty *test13_func;
  // CHECK-NEXT: [[FN:%.*]] = load void (i8*)** @test13_func, align
  // CHECK-NEXT: [[X_VAL:%.*]] = load i8** [[X]], align
  // CHECK-NEXT: [[X_TMP:%.*]] = call i8* @objc_retain(i8* [[X_VAL]]) nounwind
  // CHECK-NEXT: call void [[FN]](i8* [[X_TMP]])
  test13_func(x);

  extern fnty ^test13_block;
  // CHECK-NEXT: [[TMP:%.*]] = load void (i8*)** @test13_block, align
  // CHECK-NEXT: [[BLOCK:%.*]] = bitcast void (i8*)* [[TMP]] to [[BLOCKTY:%.*]]*
  // CHECK-NEXT: [[BLOCK_FN_PTR:%.*]] = getelementptr inbounds [[BLOCKTY]]* [[BLOCK]], i32 0, i32 3
  // CHECK-NEXT: [[BLOCK_OPAQUE:%.*]] = bitcast [[BLOCKTY]]* [[BLOCK]] to i8*
  // CHECK-NEXT: [[X_VAL:%.*]] = load i8** [[X]], align
  // CHECK-NEXT: [[X_TMP:%.*]] = call i8* @objc_retain(i8* [[X_VAL]]) nounwind
  // CHECK-NEXT: [[BLOCK_FN_TMP:%.*]] = load i8** [[BLOCK_FN_PTR]]
  // CHECK-NEXT: [[BLOCK_FN:%.*]] = bitcast i8* [[BLOCK_FN_TMP]] to void (i8*, i8*)*
  // CHECK-NEXT: call void [[BLOCK_FN]](i8* [[BLOCK_OPAQUE]], i8* [[X_TMP]])
  test13_block(x);

  // CHECK-NEXT: [[T0:%.*]] = load i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]]) nounwind
  // CHECK-NEXT: ret void
}

@interface Test16_super @end
@interface Test16 : Test16_super {
  id z;
}
@property (assign) int x;
@property (retain) id y;
- (void) dealloc;
@end
@implementation Test16
@synthesize x;
@synthesize y;
- (void) dealloc {
  // CHECK:    define internal void @"\01-[Test16 dealloc]"(
  // CHECK:      [[SELF:%.*]] = alloca [[TEST16:%.*]]*, align
  // CHECK-NEXT: [[CMD:%.*]] = alloca i8*, align
  // CHECK-NEXT: alloca
  // CHECK-NEXT: store [[TEST16]]* {{%.*}}, [[TEST16]]** [[SELF]], align
  // CHECK-NEXT: store i8* {{%.*}}, i8** [[CMD]]
  // CHECK-NEXT: [[BASE:%.*]] = load [[TEST16]]** [[SELF]]

  // Call super.
  // CHECK-NEXT: [[BASE2:%.*]] = bitcast [[TEST16]]* [[BASE]] to i8*
  // CHECK-NEXT: [[T0:%.*]] = getelementptr
  // CHECK-NEXT: store i8* [[BASE2]], i8** [[T0]]
  // CHECK-NEXT: load {{%.*}}** @"\01L_OBJC_CLASSLIST_SUP_REFS_$_
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: store
  // CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: call void bitcast (i8* ({{.*}})* @objc_msgSendSuper2 to void (
  // CHECK-NEXT: ret void
}

// .cxx_destruct
  // CHECK:    define internal void @"\01-[Test16 .cxx_destruct]"(
  // CHECK:      [[SELF:%.*]] = alloca [[TEST16:%.*]]*, align
  // CHECK-NEXT: [[CMD:%.*]] = alloca i8*, align
  // CHECK-NEXT: store [[TEST16]]* {{%.*}}, [[TEST16]]** [[SELF]], align
  // CHECK-NEXT: store i8* {{%.*}}, i8** [[CMD]]
  // CHECK-NEXT: [[BASE:%.*]] = load [[TEST16]]** [[SELF]]

  // Destroy y.
  // CHECK-NEXT: [[Y_OFF:%.*]] = load i64* @"OBJC_IVAR_$_Test16.y"
  // CHECK-NEXT: [[T0:%.*]] = bitcast [[TEST16]]* [[BASE]] to i8*
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8* [[T0]], i64 [[Y_OFF]]
  // CHECK-NEXT: [[T2:%.*]] = bitcast i8* [[T1]] to i8**
  // CHECK-NEXT: call void @objc_storeStrong(i8** [[T2]], i8* null) nounwind

  // Destroy z.
  // CHECK-NEXT: [[Z_OFF:%.*]] = load i64* @"OBJC_IVAR_$_Test16.z"
  // CHECK-NEXT: [[T0:%.*]] = bitcast [[TEST16]]* [[BASE]] to i8*
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8* [[T0]], i64 [[Z_OFF]]
  // CHECK-NEXT: [[T2:%.*]] = bitcast i8* [[T1]] to i8**
  // CHECK-NEXT: call void @objc_storeStrong(i8** [[T2]], i8* null) nounwind

  // CHECK-NEXT: ret void

@end

// This shouldn't crash.
@interface Test17A
@property (assign) int x;
@end
@interface Test17B : Test17A
@end
@implementation Test17B
- (int) x { return super.x + 1; }
@end

// This shouldn't crash.
void test18(id (^maker)(void)) {
  maker();
}

void test19() {
  // CHECK: define void @test19()
  // CHECK:      [[X:%.*]] = alloca [5 x i8*], align 16
  // CHECK-NEXT: [[T0:%.*]] = bitcast [5 x i8*]* [[X]] to i8*
  // CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* [[T0]], i8 0, i64 40, i32 16, i1 false)
  id x[5];

  extern id test19_helper(void);
  x[2] = test19_helper();

  // CHECK-NEXT: [[CALL:%.*]] = call i8* @test19_helper()
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[CALL]]) nounwind
  // CHECK-NEXT: [[SLOT:%.*]] = getelementptr inbounds [5 x i8*]* [[X]], i32 0, i64 2
  // CHECK-NEXT: [[T0:%.*]] = load i8** [[SLOT]]
  // CHECK-NEXT: store i8* [[T1]], i8** [[SLOT]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]]) nounwind

  // CHECK-NEXT: [[BEGIN:%.*]] = getelementptr inbounds [5 x i8*]* [[X]], i32 0, i32 0
  // CHECK-NEXT: [[END:%.*]] = getelementptr inbounds i8** [[BEGIN]], i64 5
  // CHECK-NEXT: br label

  // CHECK:      [[CUR:%.*]] = phi i8**
  // CHECK-NEXT: [[EQ:%.*]] = icmp eq i8** [[CUR]], [[END]]
  // CHECK-NEXT: br i1 [[EQ]],

  // CHECK:      [[T0:%.*]] = load i8** [[CUR]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]]) nounwind, !clang.imprecise_release
  // CHECK-NEXT: [[NEXT:%.*]] = getelementptr inbounds i8** [[CUR]], i32 1
  // CHECK-NEXT: br label

  // CHECK:      ret void
}

void test20(unsigned n) {
  // CHECK: define void @test20
  // CHECK:      [[N:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[SAVED_STACK:%.*]] = alloca i8*
  // CHECK-NEXT: store i32 {{%.*}}, i32* [[N]], align 4

  id x[n];

  // Capture the VLA size.
  // CHECK-NEXT: [[T0:%.*]] = load i32* [[N]], align 4
  // CHECK-NEXT: [[DIM:%.*]] = zext i32 [[T0]] to i64

  // Save the stack pointer.
  // CHECK-NEXT: [[T0:%.*]] = call i8* @llvm.stacksave()
  // CHECK-NEXT: store i8* [[T0]], i8** [[SAVED_STACK]]

  // Allocate the VLA.
  // CHECK-NEXT: [[VLA:%.*]] = alloca i8*, i64 [[DIM]], align 16

  // Zero-initialize.
  // CHECK-NEXT: [[T0:%.*]] = bitcast i8** [[VLA]] to i8*
  // CHECK-NEXT: [[T1:%.*]] = mul nuw i64 [[DIM]], 8
  // CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* [[T0]], i8 0, i64 [[T1]], i32 8, i1 false)

  // Destroy.
  // CHECK-NEXT: [[END:%.*]] = getelementptr inbounds i8** [[VLA]], i64 [[DIM]]
  // CHECK-NEXT: br label

  // CHECK:      [[CUR:%.*]] = phi i8**
  // CHECK-NEXT: [[EQ:%.*]] = icmp eq i8** [[CUR]], [[END]]
  // CHECK-NEXT: br i1 [[EQ]],

  // CHECK:      [[T0:%.*]] = load i8** [[CUR]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]]) nounwind, !clang.imprecise_release
  // CHECK-NEXT: [[NEXT:%.*]] = getelementptr inbounds i8** [[CUR]], i32 1
  // CHECK-NEXT: br label

  // CHECK:      [[T0:%.*]] = load i8** [[SAVED_STACK]]
  // CHECK-NEXT: call void @llvm.stackrestore(i8* [[T0]])
  // CHECK-NEXT: ret void
}

void test21(unsigned n) {
  // CHECK: define void @test21
  // CHECK:      [[N:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[SAVED_STACK:%.*]] = alloca i8*
  // CHECK-NEXT: store i32 {{%.*}}, i32* [[N]], align 4

  id x[2][n][3];

  // Capture the VLA size.
  // CHECK-NEXT: [[T0:%.*]] = load i32* [[N]], align 4
  // CHECK-NEXT: [[DIM:%.*]] = zext i32 [[T0]] to i64

  // CHECK-NEXT: [[T0:%.*]] = call i8* @llvm.stacksave()
  // CHECK-NEXT: store i8* [[T0]], i8** [[SAVED_STACK]]


  // Allocate the VLA.
  // CHECK-NEXT: [[T0:%.*]] = mul nuw i64 2, [[DIM]]
  // CHECK-NEXT: [[VLA:%.*]] = alloca [3 x i8*], i64 [[T0]], align 16

  // Zero-initialize.
  // CHECK-NEXT: [[T0:%.*]] = bitcast [3 x i8*]* [[VLA]] to i8*
  // CHECK-NEXT: [[T1:%.*]] = mul nuw i64 2, [[DIM]]
  // CHECK-NEXT: [[T2:%.*]] = mul nuw i64 [[T1]], 24
  // CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* [[T0]], i8 0, i64 [[T2]], i32 8, i1 false)

  // Destroy.
  // CHECK-NEXT: [[T0:%.*]] = mul nuw i64 2, [[DIM]]
  // CHECK-NEXT: [[BEGIN:%.*]] = getelementptr inbounds [3 x i8*]* [[VLA]], i32 0, i32 0
  // CHECK-NEXT: [[T1:%.*]] = mul nuw i64 [[T0]], 3
  // CHECK-NEXT: [[END:%.*]] = getelementptr inbounds i8** [[BEGIN]], i64 [[T1]]
  // CHECK-NEXT: br label

  // CHECK:      [[CUR:%.*]] = phi i8**
  // CHECK-NEXT: [[EQ:%.*]] = icmp eq i8** [[CUR]], [[END]]
  // CHECK-NEXT: br i1 [[EQ]],

  // CHECK:      [[T0:%.*]] = load i8** [[CUR]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]]) nounwind, !clang.imprecise_release
  // CHECK-NEXT: [[NEXT:%.*]] = getelementptr inbounds i8** [[CUR]], i32 1
  // CHECK-NEXT: br label

  // CHECK:      [[T0:%.*]] = load i8** [[SAVED_STACK]]
  // CHECK-NEXT: call void @llvm.stackrestore(i8* [[T0]])
  // CHECK-NEXT: ret void
}

void test22(_Bool cond) {
  id test22_helper(void) __attribute__((ns_returns_retained));

  // CHECK:      define void @test22(
  // CHECK:      [[COND:%.*]] = alloca i8,
  // CHECK-NEXT: [[X:%.*]] = alloca i8*,
  // CHECK-NEXT: [[RELCOND:%.*]] = alloca i1
  // CHECK-NEXT: [[RELVAL:%.*]] = alloca i8*
  // CHECK-NEXT: store i1 false, i1* [[RELCOND]]
  // CHECK-NEXT: zext
  // CHECK-NEXT: store
  // CHECK-NEXT: [[T0:%.*]] = load i8* [[COND]]
  // CHECK-NEXT: [[T1:%.*]] = trunc i8 [[T0]] to i1
  // CHECK-NEXT: br i1 [[T1]],
  // CHECK:      br label
  // CHECK:      [[CALL:%.*]] = call i8* @test22_helper()
  // CHECK-NEXT: store i1 true, i1* [[RELCOND]]
  // CHECK-NEXT: store i8* [[CALL]], i8** [[RELVAL]]
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
  id x = (cond ? 0 : test22_helper());
}

// rdar://problem/8922540
//   Note that we no longer emit .release_ivars flags.
// CHECK-GLOBALS: @"\01l_OBJC_CLASS_RO_$_Test23" = internal global [[RO_T:%.*]] { i32 134,
@interface Test23 { id x; } @end
@implementation Test23 @end

// CHECK-GLOBALS: @"\01l_OBJC_CLASS_RO_$_Test24" = internal global [[RO_T:%.*]] { i32 130,
@interface Test24 {} @end
@implementation Test24 @end

int (^test25(int x))(void) {
  // CHECK:    define i32 ()* @test25(
  // CHECK:      [[X:%.*]] = alloca i32,
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:%.*]],
  // CHECK-NEXT: store i32 {{%.*}}, i32* [[X]]
  // CHECK:      [[T0:%.*]] = bitcast [[BLOCK_T]]* [[BLOCK]] to i32 ()*
  // CHECK-NEXT: [[T1:%.*]] = bitcast i32 ()* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainBlock(i8* [[T1]]) nounwind
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to i32 ()*
  // CHECK-NEXT: [[T4:%.*]] = bitcast i32 ()* [[T3]] to i8*
  // CHECK-NEXT: [[T5:%.*]] = call i8* @objc_autoreleaseReturnValue(i8* [[T4]]) nounwind
  // CHECK-NEXT: [[T6:%.*]] = bitcast i8* [[T5]] to i32 ()*
  // CHECK-NEXT: ret i32 ()* [[T6]]
  return ^{ return x; };
}

// rdar://problem/8941012
@interface Test26 { id x[4]; } @end
@implementation Test26 @end
// CHECK:    define internal void @"\01-[Test26 .cxx_destruct]"(
// CHECK:      [[SELF:%.*]] = load [[TEST26:%.*]]**
// CHECK-NEXT: [[OFFSET:%.*]] = load i64* @"OBJC_IVAR_$_Test26.x"
// CHECK-NEXT: [[T0:%.*]] = bitcast [[TEST26]]* [[SELF]] to i8*
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8* [[T0]], i64 [[OFFSET]]
// CHECK-NEXT: [[X:%.*]] = bitcast i8* [[T1]] to [4 x i8*]*
// CHECK-NEXT: [[BEGIN:%.*]] = getelementptr inbounds [4 x i8*]* [[X]], i32 0, i32 0
// CHECK-NEXT: [[END:%.*]] = getelementptr inbounds i8** [[BEGIN]], i64 4
// CHECK-NEXT: br label
// CHECK:      [[CUR:%.*]] = phi i8**
// CHECK-NEXT: [[ISDONE:%.*]] = icmp eq i8** [[CUR]], [[END]]
// CHECK-NEXT: br i1 [[ISDONE]],
// CHECK:      call void @objc_storeStrong(i8** [[CUR]], i8* null)
// CHECK-NEXT: [[NEXT:%.*]] = getelementptr inbounds i8** [[CUR]], i32 1
// CHECK-NEXT: br label
// CHECK:      ret void

// Check that 'init' retains self.
@interface Test27
- (id) init;
@end
@implementation Test27
- (id) init { return self; }
// CHECK:    define internal i8* @"\01-[Test27 init]"
// CHECK:      [[RET:%.*]] = alloca i8*,
// CHECK-NEXT: [[SELF:%.*]] = alloca [[TEST27:%.*]]*,
// CHECK-NEXT: [[CMD:%.*]] = alloca i8*,
// CHECK-NEXT: [[DEST:%.*]] = alloca i32
// CHECK-NEXT: store [[TEST27]]* {{%.*}}, [[TEST27]]** [[SELF]]
// CHECK-NEXT: store i8* {{%.*}}, i8** [[CMD]]
// CHECK-NEXT: [[T0:%.*]] = load [[TEST27]]** [[SELF]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST27]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retain(i8* [[T1]])
// CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]]
// CHECK-NEXT: [[T2:%.*]] = bitcast
// CHECK-NEXT: store i8* [[T2]], i8** [[RET]]
// CHECK-NEXT: store i32 {{[0-9]+}}, i32* [[DEST]]
// CHECK-NEXT: [[T0:%.*]] = load [[TEST27]]** [[SELF]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST27]]* [[T0]] to i8*
// CHECK-NEXT: call void @objc_release(i8* [[T1]])
// CHECK-NEXT: [[T0:%.*]] = load i8** [[RET]]
// CHECK-NEXT: ret i8* [[T0]]

@end

// rdar://problem/8087194
@interface Test28
@property (copy) id prop;
@end
@implementation Test28
@synthesize prop;
@end
// CHECK:    define internal void @"\01-[Test28 .cxx_destruct]"
// CHECK:      [[SELF:%.*]] = load [[TEST28:%.*]]**
// CHECK-NEXT: [[OFFSET:%.*]] = load i64* @"OBJC_IVAR_$_Test28.prop"
// CHECK-NEXT: [[T0:%.*]] = bitcast [[TEST28]]* [[SELF]] to i8*
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8* [[T0]], i64 [[OFFSET]]
// CHECK-NEXT: [[T2:%.*]] = bitcast i8* [[T1]] to i8**
// CHECK-NEXT: call void @objc_storeStrong(i8** [[T2]], i8* null)
// CHECK-NEXT: ret void

@interface Test29_super
- (id) initWithAllocator: (id) allocator;
@end
@interface Test29 : Test29_super
- (id) init;
- (id) initWithAllocator: (id) allocator;
@end
@implementation Test29
static id _test29_allocator = 0;
- (id) init {
// CHECK:    define internal i8* @"\01-[Test29 init]"([[TEST29:%.*]]* {{%.*}},
// CHECK:      [[RET:%.*]] = alloca i8*, align 8
// CHECK-NEXT: [[SELF:%.*]] = alloca [[TEST29]]*, align 8
// CHECK-NEXT: [[CMD:%.*]] = alloca i8*, align 8
// CHECK-NEXT: [[CLEANUP:%.*]] = alloca i32
// CHECK-NEXT: store [[TEST29]]* {{%.*}}, [[TEST29]]** [[SELF]]
// CHECK-NEXT: store i8* {{%.*}}, i8** [[CMD]]

// Evaluate arguments.  Note that the send argument is evaluated
// before the zeroing of self.
// CHECK-NEXT: [[T0:%.*]] = load [[TEST29]]** [[SELF]], align 8
// CHECK-NEXT: [[T1:%.*]] = load i8** @_test29_allocator, align 8

// Implicit null of 'self', i.e. direct transfer of ownership.
// CHECK-NEXT: store [[TEST29]]* null, [[TEST29]]** [[SELF]]

// Actual message send.
// CHECK-NEXT: [[T2:%.*]] = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_
// CHECK-NEXT: [[T3:%.*]] = bitcast [[TEST29]]* [[T0]] to i8*
// CHECK-NEXT: [[CALL:%.*]] = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* [[T3]], i8* [[T2]], i8* [[T1]])

// Implicit write of result back into 'self'.  This is not supposed to
// be detectable because we're supposed to ban accesses to the old
// self value past the delegate init call.
// CHECK-NEXT: [[T0:%.*]] = bitcast i8* [[CALL]] to [[TEST29]]*
// CHECK-NEXT: store [[TEST29]]* [[T0]], [[TEST29]]** [[SELF]]

// Return statement.
// CHECK-NEXT: [[T2:%.*]] = bitcast i8* [[CALL]]
// CHECK-NEXT: [[CALL:%.*]] = bitcast
// CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retain(i8* [[CALL]]) nounwind
// CHECK-NEXT: [[T2:%.*]] = bitcast i8* [[T1]]
// CHECK-NEXT: [[T1:%.*]] = bitcast
// CHECK-NEXT: store i8* [[T1]], i8** [[RET]]
// CHECK-NEXT: store i32 1, i32* [[CLEANUP]]

// Cleanup.
// CHECK-NEXT: [[T0:%.*]] = load [[TEST29]]** [[SELF]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST29]]* [[T0]] to i8*
// CHECK-NEXT: call void @objc_release(i8* [[T1]]) nounwind, !clang.imprecise_release

// Return.
// CHECK-NEXT: [[T0:%.*]] = load i8** [[RET]]
// CHECK-NEXT: ret i8* [[T0]]
  return [self initWithAllocator: _test29_allocator];
}
- (id) initWithAllocator: (id) allocator {
// CHECK:    define internal i8* @"\01-[Test29 initWithAllocator:]"(
// CHECK:      [[RET:%.*]] = alloca i8*, align 8
// CHECK-NEXT: [[SELF:%.*]] = alloca [[TEST29]]*, align 8
// CHECK-NEXT: [[CMD:%.*]] = alloca i8*, align 8
// CHECK-NEXT: [[ALLOCATOR:%.*]] = alloca i8*, align 8
// CHECK-NEXT: alloca
// CHECK-NEXT: [[CLEANUP:%.*]] = alloca i32
// CHECK-NEXT: store [[TEST29]]* {{%.*}}, [[TEST29]]** [[SELF]]
// CHECK-NEXT: store i8* {{%.*}}, i8** [[CMD]]
// CHECK-NEXT: [[T0:%.*]] = call i8* @objc_retain(i8* {{%.*}})
// CHECK-NEXT: store i8* [[T0]], i8** [[ALLOCATOR]]

// Evaluate arguments.  Note that the send argument is evaluated
// before the zeroing of self.
// CHECK-NEXT: [[T0:%.*]] = load [[TEST29]]** [[SELF]]
// CHECK-NEXT: [[T1:%.*]] = load i8** [[ALLOCATOR]], align 8

// Implicit null of 'self', i.e. direct transfer of ownership.
// CHECK-NEXT: store [[TEST29]]* null, [[TEST29]]** [[SELF]]

// Actual message send.
// CHECK:      [[CALL:%.*]] = call {{.*}} @objc_msgSendSuper2

// Implicit write of result back into 'self'.  This is not supposed to
// be detectable because we're supposed to ban accesses to the old
// self value past the delegate init call.
// CHECK-NEXT: [[T0:%.*]] = bitcast i8* [[CALL]] to [[TEST29]]*
// CHECK-NEXT: store [[TEST29]]* [[T0]], [[TEST29]]** [[SELF]]

// Assignment.
// CHECK-NEXT: [[T0:%.*]] = bitcast i8* [[CALL]] to [[TEST29]]*
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST29]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retain(i8* [[T1]]) nounwind
// CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[TEST29]]*
// CHECK-NEXT: [[T4:%.*]] = load [[TEST29]]** [[SELF]], align
// CHECK-NEXT: store [[TEST29]]* [[T3]], [[TEST29]]** [[SELF]], align
// CHECK-NEXT: [[T5:%.*]] = bitcast [[TEST29]]* [[T4]] to i8*
// CHECK-NEXT: call void @objc_release(i8* [[T5]])

// Return statement.
// CHECK-NEXT: [[T0:%.*]] = load [[TEST29]]** [[SELF]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST29]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retain(i8* [[T1]]) nounwind
// CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]]
// CHECK-NEXT: [[T2:%.*]] = bitcast
// CHECK-NEXT: store i8* [[T2]], i8** [[RET]]
// CHECK-NEXT: store i32 1, i32* [[CLEANUP]]

// Cleanup.
// CHECK-NEXT: [[T0:%.*]] = load i8** [[ALLOCATOR]]
// CHECK-NEXT: call void @objc_release(i8* [[T0]]) nounwind, !clang.imprecise_release

// CHECK-NEXT: [[T0:%.*]] = load [[TEST29]]** [[SELF]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST29]]* [[T0]] to i8*
// CHECK-NEXT: call void @objc_release(i8* [[T1]]) nounwind, !clang.imprecise_release

// Return.
// CHECK-NEXT: [[T0:%.*]] = load i8** [[RET]]
// CHECK-NEXT: ret i8* [[T0]]
  self = [super initWithAllocator: allocator];
  return self;
}
@end

typedef struct Test30_helper Test30_helper;
@interface Test30
- (id) init;
- (Test30_helper*) initHelper;
@end
@implementation Test30 {
char *helper;
}
- (id) init {
// CHECK:    define internal i8* @"\01-[Test30 init]"([[TEST30:%.*]]* {{%.*}},
// CHECK:      [[RET:%.*]] = alloca i8*
// CHECK-NEXT: [[SELF:%.*]] = alloca [[TEST30]]*
// CHECK-NEXT: alloca i8*
// CHECK-NEXT: alloca i32
// CHECK-NEXT: store [[TEST30]]* {{%.*}}, [[TEST30]]** [[SELF]]
// CHECK-NEXT: store

// Call.
// CHECK-NEXT: [[T0:%.*]] = load [[TEST30]]** [[SELF]]
// CHECK-NEXT: [[T1:%.*]] = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_
// CHECK-NEXT: [[T2:%.*]] = bitcast [[TEST30]]* [[T0]] to i8*
// CHECK-NEXT: [[CALL:%.*]] = call [[TEST30_HELPER:%.*]]* bitcast {{.*}} @objc_msgSend {{.*}}(i8* [[T2]], i8* [[T1]])

// Assignment.
// CHECK-NEXT: [[T0:%.*]] = bitcast [[TEST30_HELPER]]* [[CALL]] to i8*
// CHECK-NEXT: [[T1:%.*]] = load [[TEST30]]** [[SELF]]
// CHECK-NEXT: [[IVAR:%.*]] = load i64* @"OBJC_IVAR_$_Test30.helper"
// CHECK-NEXT: [[T2:%.*]] = bitcast [[TEST30]]* [[T1]] to i8*
// CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds i8* [[T2]], i64 [[IVAR]]
// CHECK-NEXT: [[T4:%.*]] = bitcast i8* [[T3]] to i8**
// CHECK-NEXT#: [[T5:%.*]] = load i8** [[T4]]
// CHECK-NEXT#: [[T6:%.*]] = call i8* @objc_retain(i8* [[T0]])
// CHECK-NEXT#: call void @objc_release(i8* [[T5]])
// CHECK-NEXT: store i8* [[T0]], i8** [[T4]]

// Return.
// CHECK-NEXT: [[T0:%.*]] = load [[TEST30]]** [[SELF]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST30]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retain(i8* [[T1]])
// CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]]
// CHECK-NEXT: [[T2:%.*]] = bitcast
// CHECK-NEXT: store i8* [[T2]], i8** [[RET]]
// CHECK-NEXT: store i32 1

// Cleanup.
// CHECK-NEXT: [[T0:%.*]] = load [[TEST30]]** [[SELF]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST30]]* [[T0]] to i8*
// CHECK-NEXT: call void @objc_release(i8* [[T1]])

// Epilogue.
// CHECK-NEXT: [[T0:%.*]] = load i8** [[RET]]
// CHECK-NEXT: ret i8* [[T0]]
  self->helper = [self initHelper];
  return self;
}
- (Test30_helper*) initHelper {
// CHECK:    define internal [[TEST30_HELPER]]* @"\01-[Test30 initHelper]"(
// CHECK:      alloca
// CHECK-NEXT: alloca
// CHECK-NEXT: store
// CHECK-NEXT: store
// CHECK-NEXT: ret [[TEST30_HELPER]]* null
  return 0;
}

@end

void test31(id x) {
// CHECK:    define void @test31(
// CHECK:      [[X:%.*]] = alloca i8*,
// CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:%.*]],
// CHECK-NEXT: [[PARM:%.*]] = call i8* @objc_retain(i8* {{%.*}})
// CHECK-NEXT: store i8* [[PARM]], i8** [[X]]
// CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-NEXT: [[T0:%.*]] = load i8** [[X]],
// CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retain(i8* [[T0]])
// CHECK-NEXT: store i8* [[T1]], i8** [[SLOT]],
// CHECK-NEXT: bitcast
// CHECK-NEXT: call void @test31_helper(
// CHECK-NEXT: [[T0:%.*]] = load i8** [[SLOT]]
// CHECK-NEXT: call void @objc_release(i8* [[T0]]) nounwind, !clang.imprecise_release
// CHECK-NEXT: [[T0:%.*]] = load i8** [[X]]
// CHECK-NEXT: call void @objc_release(i8* [[T0]]) nounwind, !clang.imprecise_release
// CHECK-NEXT: ret void
  extern void test31_helper(id (^)(void));
  test31_helper(^{ return x; });
}

__attribute__((ns_returns_retained)) id test32(void) {
// CHECK:    define i8* @test32()
// CHECK:      [[CALL:%.*]] = call i8* @test32_helper()
// CHECK-NEXT: [[T0:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[CALL]])
// CHECK-NEXT: ret i8* [[T0]]
  extern id test32_helper(void);
  return test32_helper();
}

@class Test33_a;
@interface Test33
- (void) give: (Test33_a **) x;
- (void) take: (Test33_a **) x;
- (void) giveStrong: (out __strong Test33_a **) x;
- (void) takeStrong: (inout __strong Test33_a **) x;
- (void) giveOut: (out Test33_a **) x;
@end
void test33(Test33 *ptr) {
  Test33_a *a;
  [ptr give: &a];
  [ptr take: &a];
  [ptr giveStrong: &a];
  [ptr takeStrong: &a];
  [ptr giveOut: &a];

  // CHECK:    define void @test33([[TEST33:%.*]]*
  // CHECK:      [[PTR:%.*]] = alloca [[TEST33]]*
  // CHECK-NEXT: [[A:%.*]] = alloca [[A_T:%.*]]*
  // CHECK-NEXT: [[TEMP0:%.*]] = alloca [[A_T]]*
  // CHECK-NEXT: [[TEMP1:%.*]] = alloca [[A_T]]*
  // CHECK-NEXT: [[TEMP2:%.*]] = alloca [[A_T]]*
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: objc_retain
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: store
  // CHECK-NEXT: store [[A_T]]* null, [[A_T]]** [[A]]

  // CHECK-NEXT: load [[TEST33]]** [[PTR]]
  // CHECK-NEXT: [[T0:%.*]] = load [[A_T]]** [[A]]
  // CHECK-NEXT: store [[A_T]]* [[T0]], [[A_T]]** [[TEMP0]]
  // CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: objc_msgSend{{.*}}, [[A_T]]** [[TEMP0]])
  // CHECK-NEXT: [[T0:%.*]] = load [[A_T]]** [[TEMP0]]
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[A_T]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retain(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[A_T]]*
  // CHECK-NEXT: [[T4:%.*]] = load [[A_T]]** [[A]]
  // CHECK-NEXT: store [[A_T]]* [[T3]], [[A_T]]** [[A]]
  // CHECK-NEXT: [[T5:%.*]] = bitcast [[A_T]]* [[T4]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T5]])

  // CHECK-NEXT: load [[TEST33]]** [[PTR]]
  // CHECK-NEXT: [[T0:%.*]] = load [[A_T]]** [[A]]
  // CHECK-NEXT: store [[A_T]]* [[T0]], [[A_T]]** [[TEMP1]]
  // CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: objc_msgSend{{.*}}, [[A_T]]** [[TEMP1]])
  // CHECK-NEXT: [[T0:%.*]] = load [[A_T]]** [[TEMP1]]
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[A_T]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retain(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[A_T]]*
  // CHECK-NEXT: [[T4:%.*]] = load [[A_T]]** [[A]]
  // CHECK-NEXT: store [[A_T]]* [[T3]], [[A_T]]** [[A]]
  // CHECK-NEXT: [[T5:%.*]] = bitcast [[A_T]]* [[T4]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T5]])

  // CHECK-NEXT: load [[TEST33]]** [[PTR]]
  // CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: objc_msgSend{{.*}}, [[A_T]]** [[A]])

  // CHECK-NEXT: load [[TEST33]]** [[PTR]]
  // CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: objc_msgSend{{.*}}, [[A_T]]** [[A]])

  // 'out'
  // CHECK-NEXT: load [[TEST33]]** [[PTR]]
  // CHECK-NEXT: store [[A_T]]* null, [[A_T]]** [[TEMP2]]
  // CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: objc_msgSend{{.*}}, [[A_T]]** [[TEMP2]])
  // CHECK-NEXT: [[T0:%.*]] = load [[A_T]]** [[TEMP2]]
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[A_T]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retain(i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[A_T]]*
  // CHECK-NEXT: [[T4:%.*]] = load [[A_T]]** [[A]]
  // CHECK-NEXT: store [[A_T]]* [[T3]], [[A_T]]** [[A]]
  // CHECK-NEXT: [[T5:%.*]] = bitcast [[A_T]]* [[T4]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T5]])

  // CHECK-NEXT: load
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: objc_release
  // CHECK-NEXT: load
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: objc_release
  // CHECK-NEXT: ret void
}

void test34(int cond) {
  __strong id strong;
  __weak id weak;
  extern void test34_sink(id *);
  test34_sink(cond ? &strong : 0);
  test34_sink(cond ? &weak : 0);

  // CHECK:    define void @test34(
  // CHECK:      [[COND:%.*]] = alloca i32
  // CHECK-NEXT: [[STRONG:%.*]] = alloca i8*
  // CHECK-NEXT: [[WEAK:%.*]] = alloca i8*
  // CHECK-NEXT: [[TEMP1:%.*]] = alloca i8*
  // CHECK-NEXT: [[TEMP2:%.*]] = alloca i8*
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
  // CHECK:      call void @test34_sink(i8** [[T1]])
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
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = call i8* @objc_loadWeak(i8** [[ARG]])
  // CHECK-NEXT: store i8* [[T0]], i8** [[TEMP2]]
  // CHECK-NEXT: br label
  // CHECK:      call void @test34_sink(i8** [[T1]])
  // CHECK-NEXT: [[T0:%.*]] = icmp eq i8** [[ARG]], null
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = load i8** [[TEMP2]]
  // CHECK-NEXT: call i8* @objc_storeWeak(i8** [[ARG]], i8* [[T0]])
  // CHECK-NEXT: br label

  // CHECK:      call void @objc_destroyWeak(i8** [[WEAK]])
  // CHECK:      ret void
}

void test35(void (^sink)(id*)) {
  __strong id strong;
  sink(&strong);

  // CHECK:    define void @test35(
  // CHECK:      [[SINK:%.*]] = alloca void (i8**)*
  // CHECK-NEXT: [[STRONG:%.*]] = alloca i8*
  // CHECK-NEXT: [[TEMP:%.*]] = alloca i8*
  // CHECK-NEXT: bitcast void (i8**)* {{%.*}} to i8*
  // CHECK-NEXT: call i8* @objc_retain(
  // CHECK-NEXT: bitcast i8*
  // CHECK-NEXT: store void (i8**)* {{%.*}}, void (i8**)** [[SINK]]
  // CHECK-NEXT: store i8* null, i8** [[STRONG]]

  // CHECK-NEXT: load void (i8**)** [[SINK]]
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: [[BLOCK:%.*]] = bitcast
  // CHECK-NEXT: [[T0:%.*]] = load i8** [[STRONG]]
  // CHECK-NEXT: store i8* [[T0]], i8** [[TEMP1]]
  // CHECK-NEXT: [[F0:%.*]] = load i8**
  // CHECK-NEXT: [[F1:%.*]] = bitcast i8* [[F0]] to void (i8*, i8**)*
  // CHECK-NEXT: call void [[F1]](i8* [[BLOCK]], i8** [[TEMP1]])
  // CHECK-NEXT: [[T0:%.*]] = load i8** [[TEMP1]]
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retain(i8* [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = load i8** [[STRONG]]
  // CHECK-NEXT: store i8* [[T1]], i8** [[STRONG]]
  // CHECK-NEXT: call void @objc_release(i8* [[T2]])

  // CHECK-NEXT: [[T0:%.*]] = load i8** [[STRONG]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]])

  // CHECK-NEXT: load void (i8**)** [[SINK]]
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: call void @objc_release
  // CHECK-NEXT: ret void

}

// CHECK: define void @test36
void test36(id x) {
  // CHECK: [[X:%.*]] = alloca i8*

  // CHECK: call i8* @objc_retain
  // CHECK: call i8* @objc_retain
  // CHECK: call i8* @objc_retain
  id array[3] = { @"A", x, @"y" };

  // CHECK:      [[T0:%.*]] = load i8** [[X]]
  // CHECK-NEXT: store i8* null, i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]])
  x = 0;

  // CHECK: br label
  // CHECK: call void @objc_release
  // CHECK: br label

  // CHECK: call void @objc_release
  // CHECK-NEXT: ret void
}

@class Test37;
void test37(void) {
  extern void test37_helper(id *);
  Test37 *var;
  test37_helper(&var);

  // CHECK:    define void @test37()
  // CHECK:      [[VAR:%.*]] = alloca [[TEST37:%.*]]*,
  // CHECK-NEXT: [[TEMP:%.*]] = alloca i8*
  // CHECK-NEXT: store [[TEST37]]* null, [[TEST37]]** [[VAR]]

  // CHECK-NEXT: [[T0:%.*]] = load [[TEST37]]** [[VAR]]
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST37]]* [[T0]] to i8*
  // CHECK-NEXT: store i8* [[T1]], i8** [[TEMP]]
  // CHECK-NEXT: call void @test37_helper(i8** [[TEMP]])
  // CHECK-NEXT: [[T0:%.*]] = load i8** [[TEMP]]
  // CHECK-NEXT: [[T1:%.*]] = bitcast i8* [[T0]] to [[TEST37]]*
  // CHECK-NEXT: [[T2:%.*]] = bitcast [[TEST37]]* [[T1]] to i8*
  // CHECK-NEXT: [[T3:%.*]] = call i8* @objc_retain(i8* [[T2]])
  // CHECK-NEXT: [[T4:%.*]] = bitcast i8* [[T3]] to [[TEST37]]*
  // CHECK-NEXT: [[T5:%.*]] = load [[TEST37]]** [[VAR]]
  // CHECK-NEXT: store [[TEST37]]* [[T4]], [[TEST37]]** [[VAR]]
  // CHECK-NEXT: [[T6:%.*]] = bitcast [[TEST37]]* [[T5]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T6]])

  // CHECK-NEXT: [[T0:%.*]] = load [[TEST37]]** [[VAR]]
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST37]]* [[T0]] to i8*
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])
  // CHECK-NEXT: ret void
}

void test38(void) {
  id test38_source(void);
  void test38_helper(void (^)(void));
  __block id var = test38_source();
  test38_helper(^{ var = 0; });

  // CHECK:    define void @test38()
  // CHECK:      [[VAR:%.*]] = alloca [[BYREF_T:%.*]],
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:%.*]],
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]]* [[VAR]], i32 0, i32 2
  // 0x02000000 - has copy/dispose helpers
  // CHECK-NEXT: store i32 33554432, i32* [[T0]]
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]]* [[VAR]], i32 0, i32 6
  // CHECK-NEXT: [[T0:%.*]] = call i8* @test38_source()
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: store i8* [[T1]], i8** [[SLOT]]
  // CHECK-NEXT: [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]]* [[VAR]], i32 0, i32 6
  // 0x42000000 - has signature, copy/dispose helpers
  // CHECK:      store i32 1107296256,
  // CHECK:      [[T0:%.*]] = bitcast [[BYREF_T]]* [[VAR]] to i8*
  // CHECK-NEXT: store i8* [[T0]], i8**
  // CHECK:      call void @test38_helper(
  // CHECK:      [[T0:%.*]] = bitcast [[BYREF_T]]* [[VAR]] to i8*
  // CHECK-NEXT: call void @_Block_object_dispose(i8* [[T0]], i32 8)
  // CHECK-NEXT: [[T0:%.*]] = load i8** [[SLOT]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]])
  // CHECK-NEXT: ret void

  // CHECK:    define internal void @__Block_byref_object_copy_
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]]* {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: load i8**
  // CHECK-NEXT: bitcast i8* {{%.*}} to [[BYREF_T]]*
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[BYREF_T]]* {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: [[T2:%.*]] = load i8** [[T1]]
  // CHECK-NEXT: store i8* [[T2]], i8** [[T0]]
  // CHECK-NEXT: store i8* null, i8** [[T1]]

  // CHECK:    define internal void @__Block_byref_object_dispose_
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]]* {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: [[T1:%.*]] = load i8** [[T0]]
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])

  // CHECK:    define internal void @__test38_block_invoke_
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]]* {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: [[T0:%.*]] = load i8** [[SLOT]], align 8
  // CHECK-NEXT: store i8* null, i8** [[SLOT]],
  // CHECK-NEXT: call void @objc_release(i8* [[T0]])
  // CHECK-NEXT: ret void

  // CHECK:    define internal void @__copy_helper_block_
  // CHECK:      call void @_Block_object_assign(i8* {{%.*}}, i8* {{%.*}}, i32 8)

  // CHECK:    define internal void @__destroy_helper_block_
  // CHECK:      call void @_Block_object_dispose(i8* {{%.*}}, i32 8)
}

void test39(void) {
  extern id test39_source(void);
  void test39_helper(void (^)(void));
  __unsafe_unretained id var = test39_source();
  test39_helper(^{ (void) var; });

  // CHECK:    define void @test39()
  // CHECK:      [[VAR:%.*]] = alloca i8*
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:%.*]],
  // CHECK-NEXT: [[T0:%.*]] = call i8* @test39_source()
  // CHECK-NEXT: store i8* [[T0]], i8** [[VAR]],
  // 0x40000000 - has signature but no copy/dispose
  // CHECK:      store i32 1073741824, i32*
  // CHECK:      [[CAPTURE:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
  // CHECK-NEXT: [[T0:%.*]] = load i8** [[VAR]]
  // CHECK-NEXT: store i8* [[T0]], i8** [[CAPTURE]]
  // CHECK-NEXT: [[T0:%.*]] = bitcast [[BLOCK_T]]* [[BLOCK]] to void ()*
  // CHECK-NEXT: call void @test39_helper(void ()* [[T0]])
  // CHECK-NEXT: ret void
}

void test40(void) {
  id test40_source(void);
  void test40_helper(void (^)(void));
  __block __weak id var = test40_source();
  test40_helper(^{ var = 0; });

  // CHECK:    define void @test40()
  // CHECK:      [[VAR:%.*]] = alloca [[BYREF_T:%.*]],
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:%.*]],
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]]* [[VAR]], i32 0, i32 2
  // 0x02000000 - has copy/dispose helpers
  // CHECK-NEXT: store i32 33554432, i32* [[T0]]
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]]* [[VAR]], i32 0, i32 6
  // CHECK-NEXT: [[T0:%.*]] = call i8* @test40_source()
  // CHECK-NEXT: call i8* @objc_initWeak(i8** [[SLOT]], i8* [[T0]])
  // CHECK-NEXT: [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]]* [[VAR]], i32 0, i32 6
  // 0x42000000 - has signature, copy/dispose helpers
  // CHECK:      store i32 1107296256,
  // CHECK:      [[T0:%.*]] = bitcast [[BYREF_T]]* [[VAR]] to i8*
  // CHECK-NEXT: store i8* [[T0]], i8**
  // CHECK:      call void @test40_helper(
  // CHECK:      [[T0:%.*]] = bitcast [[BYREF_T]]* [[VAR]] to i8*
  // CHECK-NEXT: call void @_Block_object_dispose(i8* [[T0]], i32 8)
  // CHECK-NEXT: call void @objc_destroyWeak(i8** [[SLOT]])
  // CHECK-NEXT: ret void

  // CHECK:    define internal void @__Block_byref_object_copy_
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]]* {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: load i8**
  // CHECK-NEXT: bitcast i8* {{%.*}} to [[BYREF_T]]*
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[BYREF_T]]* {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: call void @objc_moveWeak(i8** [[T0]], i8** [[T1]])

  // CHECK:    define internal void @__Block_byref_object_dispose_
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_T]]* {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: call void @objc_destroyWeak(i8** [[T0]])

  // CHECK:    define internal void @__test40_block_invoke_
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BYREF_T]]* {{%.*}}, i32 0, i32 6
  // CHECK-NEXT: call i8* @objc_storeWeak(i8** [[SLOT]], i8* null)
  // CHECK-NEXT: ret void

  // CHECK:    define internal void @__copy_helper_block_
  // 0x8 - FIELD_IS_BYREF (no FIELD_IS_WEAK because clang in control)
  // CHECK:      call void @_Block_object_assign(i8* {{%.*}}, i8* {{%.*}}, i32 8)

  // CHECK:    define internal void @__destroy_helper_block_
  // 0x8 - FIELD_IS_BYREF (no FIELD_IS_WEAK because clang in control)
  // CHECK:      call void @_Block_object_dispose(i8* {{%.*}}, i32 8)
}

void test41(void) {
  id test41_source(void);
  void test41_helper(void (^)(void));
  void test41_consume(id);
  __weak id var = test41_source();
  test41_helper(^{ test41_consume(var); });

  // CHECK:    define void @test41()
  // CHECK:      [[VAR:%.*]] = alloca i8*,
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:%.*]],
  // CHECK:      [[T0:%.*]] = call i8* @test41_source()
  // CHECK-NEXT: call i8* @objc_initWeak(i8** [[VAR]], i8* [[T0]])
  // 0x42000000 - has signature, copy/dispose helpers
  // CHECK:      store i32 1107296256,
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
  // CHECK-NEXT: [[T0:%.*]] = call i8* @objc_loadWeak(i8** [[VAR]])
  // CHECK-NEXT: call i8* @objc_initWeak(i8** [[SLOT]], i8* [[T0]])
  // CHECK:      call void @test41_helper(
  // CHECK-NEXT: call void @objc_destroyWeak(i8** [[SLOT]])
  // CHECK-NEXT: call void @objc_destroyWeak(i8** [[VAR]])
  // CHECK-NEXT: ret void

  // CHECK:    define internal void @__test41_block_invoke_
  // CHECK:      [[SLOT:%.*]] = getelementptr inbounds [[BLOCK_T]]* {{%.*}}, i32 0, i32 5
  // CHECK-NEXT: [[T0:%.*]] = call i8* @objc_loadWeak(i8** [[SLOT]])
  // CHECK-NEXT: call void @test41_consume(i8* [[T0]])
  // CHECK-NEXT: ret void

  // CHECK:    define internal void @__copy_helper_block_
  // CHECK:      getelementptr
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: call void @objc_copyWeak(

  // CHECK:    define internal void @__destroy_helper_block_
  // CHECK:      getelementptr
  // CHECK-NEXT: call void @objc_destroyWeak(
}

@interface Test42 @end
@implementation Test42
- (void) test {
// CHECK:    define internal void @"\01-[Test42 test]"
// CHECK:      [[SELF:%.*]] = alloca [[TEST42:%.*]]*,
// CHECK-NEXT: alloca i8*
// CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:%.*]],
// CHECK-NEXT: store
// CHECK-NEXT: store
// CHECK:      [[T0:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-NEXT: [[T1:%.*]] = load [[TEST42]]** [[SELF]],
// CHECK-NEXT: [[T2:%.*]] = bitcast [[TEST42]]* [[T1]] to i8*
// CHECK-NEXT: [[T3:%.*]] = call i8* @objc_retain(i8* [[T2]])
// CHECK-NEXT: [[T4:%.*]] = bitcast i8* [[T3]] to [[TEST42]]*
// CHECK-NEXT: store [[TEST42]]* [[T4]], [[TEST42]]** [[T0]]
// CHECK-NEXT: bitcast [[BLOCK_T]]* [[BLOCK]] to void ()*
// CHECK-NEXT: call void @test42_helper(
// CHECK-NEXT: [[T1:%.*]] = load [[TEST42]]** [[T0]]
// CHECK-NEXT: [[T2:%.*]] = bitcast [[TEST42]]* [[T1]] to i8*
// CHECK-NEXT: call void @objc_release(i8* [[T2]])
// CHECK-NEXT: ret void

  extern void test42_helper(void (^)(void));
  test42_helper(^{ (void) self; });
}
@end

@interface Test43 @end
@implementation Test43
- (id) test __attribute__((ns_returns_retained)) {
  extern id test43_produce(void);
  return test43_produce();
  // CHECK:      call i8* @test43_produce()
  // CHECK-NEXT: call i8* @objc_retainAutoreleasedReturnValue(
  // CHECK-NEXT: ret 
}
@end

id test44(void) {
  typedef id __attribute__((ns_returns_retained)) blocktype(void);
  extern test44_consume_block(blocktype^);
  return ^blocktype {
      extern id test44_produce(void);
      return test44_produce();
  }();

// CHECK:    define i8* @test44(
// CHECK:      load i8** getelementptr
// CHECK-NEXT: bitcast i8*
// CHECK-NEXT: call i8* 
// CHECK-NEXT: call i8* @objc_autoreleaseReturnValue
// CHECK-NEXT: ret i8*

// CHECK:      call i8* @test44_produce()
// CHECK-NEXT: call i8* @objc_retain
// CHECK-NEXT: ret i8*
}

@interface Test45
@property (retain) id x;
@end
@implementation Test45
@synthesize x;
@end
// CHECK:    define internal i8* @"\01-[Test45 x]"(
// CHECK:      [[CALL:%.*]] = call i8* @objc_getProperty(
// CHECK-NEXT: ret i8* [[CALL]]

// rdar://problem/9315552
void test46(__weak id *wp, __weak volatile id *wvp) {
  extern id test46_helper(void);

  // TODO: this is sub-optimal, we should retain at the actual call site.

  // CHECK:      [[T0:%.*]] = call i8* @test46_helper()
  // CHECK-NEXT: [[T1:%.*]] = load i8*** {{%.*}}, align 8
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_storeWeak(i8** [[T1]], i8* [[T0]])
  // CHECK-NEXT: [[T3:%.*]] = call i8* @objc_retain(i8* [[T2]])
  // CHECK-NEXT: store i8* [[T3]], i8**
  id x = *wp = test46_helper();

  // CHECK:      [[T0:%.*]] = call i8* @test46_helper()
  // CHECK-NEXT: [[T1:%.*]] = load i8*** {{%.*}}, align 8
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_storeWeak(i8** [[T1]], i8* [[T0]])
  // CHECK-NEXT: [[T3:%.*]] = call i8* @objc_retain(i8* [[T2]])
  // CHECK-NEXT: store i8* [[T3]], i8**
  id y = *wvp = test46_helper();
}

// rdar://problem/9378887
void test47(void) {
  extern id test47_helper(void);
  id x = x = test47_helper();

  // CHECK:    define void @test47()
  // CHECK:      [[X:%.*]] = alloca i8*
  // CHECK-NEXT: store i8* null, i8** [[X]]
  // CHECK-NEXT: [[CALL:%.*]] = call i8* @test47_helper()
  // CHECK-NEXT: [[T0:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[CALL]])
  // CHECK-NEXT: [[T1:%.*]] = load i8** [[X]]
  // CHECK-NEXT: store i8* [[T0]], i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retain(i8* [[T0]])
  // CHECK-NEXT: [[T3:%.*]] = load i8** [[X]]
  // CHECK-NEXT: store i8* [[T2]], i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T3]])
  // CHECK-NEXT: [[T4:%.*]] = load i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[T4]])
  // CHECK-NEXT: ret void
}

void test48(void) {
  extern id test48_helper(void);
  __weak id x = x = test48_helper();
  // CHECK:    define void @test48()
  // CHECK:      [[X:%.*]] = alloca i8*
  // CHECK-NEXT: [[T0:%.*]] = call i8* @objc_initWeak(i8** [[X]], i8* null)
  // CHECK-NEXT: [[T1:%.*]] = call i8* @test48_helper()
  // CHECK-NEXT: [[T2:%.*]] = call i8* @objc_storeWeak(i8** [[X]], i8* [[T1]])
  // CHECK-NEXT: [[T3:%.*]] = call i8* @objc_storeWeak(i8** [[X]], i8* [[T2]])
  // CHECK-NEXT: call void @objc_destroyWeak(i8** [[X]])
  // CHECK-NEXT: ret void
}

void test49(void) {
  extern id test49_helper(void);
  __autoreleasing id x = x = test49_helper();
  // CHECK:    define void @test49()
  // CHECK:      [[X:%.*]] = alloca i8*
  // CHECK-NEXT: store i8* null, i8** [[X]]
  // CHECK-NEXT: [[CALL:%.*]] = call i8* @test49_helper()
  // CHECK-NEXT: [[T0:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[CALL]])
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_autorelease(i8* [[T0]])
  // CHECK-NEXT: store i8* [[T2]], i8** [[X]]
  // CHECK-NEXT: [[T3:%.*]] = call i8* @objc_retainAutorelease(i8* [[T1]])
  // CHECK-NEXT: store i8* [[T3]], i8** [[X]]
  // CHECK-NEXT: ret void
}

// rdar://9380136
id x();
void test50(id y) {
  ({x();});
// CHECK: [[T0:%.*]] = call i8* @objc_retain
// CHECK: call void @objc_release
}


// rdar://9400762
struct CGPoint {
  float x;
  float y;
};
typedef struct CGPoint CGPoint;

@interface Foo
@property (assign) CGPoint point;
@end

@implementation Foo
@synthesize point;
@end

// rdar://problem/9400398
id test52(void) {
  id test52_helper(int) __attribute__((ns_returns_retained));
  return ({ int x = 5; test52_helper(x); });

// CHECK:    define i8* @test52()
// CHECK:      [[X:%.*]] = alloca i32
// CHECK-NEXT: store i32 5, i32* [[X]],
// CHECK-NEXT: [[T0:%.*]] = load i32* [[X]],
// CHECK-NEXT: [[T1:%.*]] = call i8* @test52_helper(i32 [[T0]])
// CHECK-NEXT: [[T2:%.*]] = call i8* @objc_autoreleaseReturnValue(i8* [[T1]])
// CHECK-NEXT: ret i8* [[T2]]
}

// rdar://problem/9400644
void test53(void) {
  id test53_helper(void);
  id x = ({ id y = test53_helper(); y; });
  (void) x;
// CHECK:    define void @test53()
// CHECK:      [[X:%.*]] = alloca i8*,
// CHECK-NEXT: [[Y:%.*]] = alloca i8*,
// CHECK-NEXT: [[T0:%.*]] = call i8* @test53_helper()
// CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
// CHECK-NEXT: store i8* [[T1]], i8** [[Y]],
// CHECK-NEXT: [[T0:%.*]] = load i8** [[Y]],
// CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retain(i8* [[T0]])
// CHECK-NEXT: [[T2:%.*]] = load i8** [[Y]]
// CHECK-NEXT: call void @objc_release(i8* [[T2]])
// CHECK-NEXT: store i8* [[T1]], i8** [[X]],
// CHECK-NEXT: load i8** [[X]],
// CHECK-NEXT: [[T0:%.*]] = load i8** [[X]]
// CHECK-NEXT: call void @objc_release(i8* [[T0]])
// CHECK-NEXT: ret void
}
