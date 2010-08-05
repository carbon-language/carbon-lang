// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -ftrapv %s -emit-llvm -o - | FileCheck %s

// CHECK: [[I32O:%.*]] = type { i32, i1 }

unsigned int ui, uj, uk;
int i, j, k;

// CHECK: define void @test0()
void test0() {
  // -ftrapv doesn't affect unsigned arithmetic.
  // CHECK:      [[T1:%.*]] = load i32* @uj
  // CHECK-NEXT: [[T2:%.*]] = load i32* @uk
  // CHECK-NEXT: [[T3:%.*]] = add i32 [[T1]], [[T2]]
  // CHECK-NEXT: store i32 [[T3]], i32* @ui
  ui = uj + uk;

  // CHECK:      [[T1:%.*]] = load i32* @j
  // CHECK-NEXT: [[T2:%.*]] = load i32* @k
  // CHECK-NEXT: [[T3:%.*]] = call [[I32O]] @llvm.sadd.with.overflow.i32(i32 [[T1]], i32 [[T2]])
  // CHECK-NEXT: [[T4:%.*]] = extractvalue [[I32O]] [[T3]], 0
  // CHECK-NEXT: [[T5:%.*]] = extractvalue [[I32O]] [[T3]], 1
  // CHECK-NEXT: br i1 [[T5]]
  // CHECK:      [[F:%.*]] = load i64 (i64, i64, i8, i8)** @__overflow_handler
  // CHECK-NEXT: [[T6:%.*]] = sext i32 [[T1]] to i64
  // CHECK-NEXT: [[T7:%.*]] = sext i32 [[T2]] to i64
  // CHECK-NEXT: [[T8:%.*]] = call i64 [[F]](i64 [[T6]], i64 [[T7]], i8 3, i8 32)
  // CHECK-NEXT: [[T9:%.*]] = trunc i64 [[T8]] to i32
  // CHECK-NEXT: br label
  // CHECK:      [[T10:%.*]] = phi i32 [ [[T4]], {{.*}} ], [ [[T9]], {{.*}} ]
  // CHECK-NEXT: store i32 [[T10]], i32* @i
  i = j + k;
}

// CHECK: define void @test1()
void test1() {
  extern void opaque(int);
  opaque(i++);

  // CHECK:      [[T1:%.*]] = load i32* @i
  // CHECK-NEXT: [[T2:%.*]] = call [[I32O]] @llvm.sadd.with.overflow.i32(i32 [[T1]], i32 1)
  // CHECK-NEXT: [[T3:%.*]] = extractvalue [[I32O]] [[T2]], 0
  // CHECK-NEXT: [[T4:%.*]] = extractvalue [[I32O]] [[T2]], 1
  // CHECK-NEXT: br i1 [[T4]]
  // CHECK:      [[F:%.*]] = load i64 (i64, i64, i8, i8)** @__overflow_handler
  // CHECK-NEXT: [[T5:%.*]] = sext i32 [[T1]] to i64
  // CHECK-NEXT: [[T6:%.*]] = call i64 [[F]](i64 [[T5]], i64 1, i8 3, i8 32)
  // CHECK-NEXT: [[T7:%.*]] = trunc i64 [[T6]] to i32
  // CHECK-NEXT: br label
  // CHECK:      [[T8:%.*]] = phi i32 [ [[T3]], {{.*}} ], [ [[T7]], {{.*}} ]
  // CHECK-NEXT: store i32 [[T8]], i32* @i
  // CHECK-NEXT: call void @opaque(i32 [[T1]])
}

// CHECK: define void @test2()
void test2() {
  extern void opaque(int);
  opaque(++i);

  // CHECK:      [[T1:%.*]] = load i32* @i
  // CHECK-NEXT: [[T2:%.*]] = call [[I32O]] @llvm.sadd.with.overflow.i32(i32 [[T1]], i32 1)
  // CHECK-NEXT: [[T3:%.*]] = extractvalue [[I32O]] [[T2]], 0
  // CHECK-NEXT: [[T4:%.*]] = extractvalue [[I32O]] [[T2]], 1
  // CHECK-NEXT: br i1 [[T4]]
  // CHECK:      [[F:%.*]] = load i64 (i64, i64, i8, i8)** @__overflow_handler
  // CHECK-NEXT: [[T5:%.*]] = sext i32 [[T1]] to i64
  // CHECK-NEXT: [[T6:%.*]] = call i64 [[F]](i64 [[T5]], i64 1, i8 3, i8 32)
  // CHECK-NEXT: [[T7:%.*]] = trunc i64 [[T6]] to i32
  // CHECK-NEXT: br label
  // CHECK:      [[T8:%.*]] = phi i32 [ [[T3]], {{.*}} ], [ [[T7]], {{.*}} ]
  // CHECK-NEXT: store i32 [[T8]], i32* @i
  // CHECK-NEXT: call void @opaque(i32 [[T8]])
}
