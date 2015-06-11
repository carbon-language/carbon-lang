// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s | FileCheck %s

// Verify assume_safety vectorization is recognized.
void vectorize_test(int *List, int Length) {
// CHECK: define {{.*}} @_Z14vectorize_testPii
// CHECK: [[LOAD1_IV:.+]] = load i32, i32* [[IV1:[^,]+]], {{.*}}!llvm.mem.parallel_loop_access ![[LOOP1_ID:[0-9]+]]
// CHECK-NEXT: [[LOAD1_LEN:.+]] = load i32, i32* [[LEN1:.+]], {{.*}}!llvm.mem.parallel_loop_access ![[LOOP1_ID]]
// CHECK-NEXT: [[CMP1:.+]] = icmp slt i32[[LOAD1_IV]],[[LOAD1_LEN]]
// CHECK-NEXT: br i1[[CMP1]], label %[[LOOP1_BODY:[^,]+]], label %[[LOOP1_END:[^,]+]], !llvm.loop ![[LOOP1_HINTS:[0-9]+]]
#pragma clang loop vectorize(assume_safety)
  for (int i = 0; i < Length; i++) {
    // CHECK: [[LOOP1_BODY]]
    // CHECK-NEXT: [[RHIV1:.+]] = load i32, i32* [[IV1]], {{.*}}!llvm.mem.parallel_loop_access ![[LOOP1_ID]]
    // CHECK-NEXT: [[CALC1:.+]] = mul nsw i32[[RHIV1]], 2
    // CHECK-NEXT: [[SIV1:.+]] = load i32, i32* [[IV1]]{{.*}}!llvm.mem.parallel_loop_access ![[LOOP1_ID]]
    // CHECK-NEXT: [[INDEX1:.+]] = sext i32[[SIV1]] to i64
    // CHECK-NEXT: [[ARRAY1:.+]] = load i32*, i32** [[LIST1:.*]], {{.*}}!llvm.mem.parallel_loop_access ![[LOOP1_ID]]
    // CHECK-NEXT: [[PTR1:.+]] = getelementptr inbounds i32, i32*[[ARRAY1]], i64[[INDEX1]]
    // CHECK-NEXT: store i32[[CALC1]], i32*[[PTR1]], {{.*}}!llvm.mem.parallel_loop_access ![[LOOP1_ID]]
    List[i] = i * 2;
  }
  // CHECK: [[LOOP1_END]]
}

// Verify assume_safety interleaving is recognized.
void interleave_test(int *List, int Length) {
// CHECK: define {{.*}} @_Z15interleave_testPii
// CHECK: [[LOAD2_IV:.+]] = load i32, i32* [[IV2:[^,]+]], {{.*}}!llvm.mem.parallel_loop_access ![[LOOP2_ID:[0-9]+]]
// CHECK-NEXT: [[LOAD2_LEN:.+]] = load i32, i32* [[LEN2:.+]], {{.*}}!llvm.mem.parallel_loop_access ![[LOOP2_ID]]
// CHECK-NEXT: [[CMP2:.+]] = icmp slt i32[[LOAD2_IV]],[[LOAD2_LEN]]
// CHECK-NEXT: br i1[[CMP2]], label %[[LOOP2_BODY:[^,]+]], label %[[LOOP2_END:[^,]+]], !llvm.loop ![[LOOP2_HINTS:[0-9]+]]
#pragma clang loop interleave(assume_safety)
  for (int i = 0; i < Length; i++) {
    // CHECK: [[LOOP2_BODY]]
    // CHECK-NEXT: [[RHIV2:.+]] = load i32, i32* [[IV2]], {{.*}}!llvm.mem.parallel_loop_access ![[LOOP2_ID]]
    // CHECK-NEXT: [[CALC2:.+]] = mul nsw i32[[RHIV2]], 2
    // CHECK-NEXT: [[SIV2:.+]] = load i32, i32* [[IV2]]{{.*}}!llvm.mem.parallel_loop_access ![[LOOP2_ID]]
    // CHECK-NEXT: [[INDEX2:.+]] = sext i32[[SIV2]] to i64
    // CHECK-NEXT: [[ARRAY2:.+]] = load i32*, i32** [[LIST2:.*]], {{.*}}!llvm.mem.parallel_loop_access ![[LOOP2_ID]]
    // CHECK-NEXT: [[PTR2:.+]] = getelementptr inbounds i32, i32*[[ARRAY2]], i64[[INDEX2]]
    // CHECK-NEXT: store i32[[CALC2]], i32*[[PTR2]], {{.*}}!llvm.mem.parallel_loop_access ![[LOOP2_ID]]
    List[i] = i * 2;
  }
  // CHECK: [[LOOP2_END]]
}

// CHECK: ![[LOOP1_HINTS]] = distinct !{![[LOOP1_HINTS]], ![[INTENABLE_1:.*]]}
// CHCCK: ![[INTENABLE_1]] = !{!"llvm.loop.vectorize.enable", i1 true}
// CHECK: ![[LOOP2_HINTS]] = distinct !{![[LOOP2_HINTS]], ![[INTENABLE_1:.*]]}
