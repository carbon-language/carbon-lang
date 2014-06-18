// RUN: %clang_cc1 -triple i686--windows -fms-compatibility -Oz -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple thumbv7--windows -fms-compatibility -Oz -emit-llvm %s -o - | FileCheck %s

void *test_InterlockedExchangePointer(void * volatile *Target, void *Value) {
  return _InterlockedExchangePointer(Target, Value);
}

// CHECK: define{{.*}}i8* @test_InterlockedExchangePointer(i8** %Target, i8* %Value){{.*}}{
// CHECK:   %[[TARGET:[0-9]+]] = bitcast i8** %Target to i32*
// CHECK:   %[[VALUE:[0-9]+]] = ptrtoint i8* %Value to i32
// CHECK:   %[[EXCHANGE:[0-9]+]] = atomicrmw xchg i32* %[[TARGET]], i32 %[[VALUE]] seq_cst
// CHECK:   %[[RESULT:[0-9]+]] = inttoptr i32 %[[EXCHANGE]] to i8*
// CHECK:   ret i8* %[[RESULT]]
// CHECK: }

void *test_InterlockedCompareExchangePointer(void * volatile *Destination,
                                             void *Exchange, void *Comparand) {
  return _InterlockedCompareExchangePointer(Destination, Exchange, Comparand);
}

// CHECK: define{{.*}}i8* @test_InterlockedCompareExchangePointer(i8** %Destination, i8* %Exchange, i8* %Comparand){{.*}}{
// CHECK:   %[[DEST:[0-9]+]] = bitcast i8** %Destination to i32*
// CHECK:   %[[EXCHANGE:[0-9]+]] = ptrtoint i8* %Exchange to i32
// CHECK:   %[[COMPARAND:[0-9]+]] = ptrtoint i8* %Comparand to i32
// CHECK:   %[[XCHG:[0-9]+]] = cmpxchg volatile i32* %[[DEST:[0-9]+]], i32 %[[COMPARAND:[0-9]+]], i32 %[[EXCHANGE:[0-9]+]] seq_cst seq_cst
// CHECK:   %[[EXTRACT:[0-9]+]] = extractvalue { i32, i1 } %[[XCHG]], 0
// CHECK:   %[[RESULT:[0-9]+]] = inttoptr i32 %[[EXTRACT]] to i8*
// CHECK:   ret i8* %[[RESULT:[0-9]+]]
// CHECK: }

long test_InterlockedExchange(long *Target, long Value) {
  return _InterlockedExchange(Target, Value);
}

// CHECK: define{{.*}}i32 @test_InterlockedExchange(i32* %Target, i32 %Value){{.*}}{
// CHECK:   %[[EXCHANGE:[0-9]+]] = atomicrmw xchg i32* %Target, i32 %Value seq_cst
// CHECK:   ret i32 %[[EXCHANGE:[0-9]+]]
// CHECK: }
